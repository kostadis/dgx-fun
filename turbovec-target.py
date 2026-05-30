#!/usr/bin/env python3
"""turbovec hybrid — short-query / known-target retrieval, DETERMINISTIC queries.
Runs on the Spark, vllm-embed on :8000. No LLM in the test substrate.

Fixes the circular benchmark in turbovec-hybrid.py (gold = exact-vector top-k,
so BM25 could only hurt). Here:

  * For K held-out drawers (the TARGETS), build a short keyword query from the
    drawer's most DISTINCTIVE terms (high corpus-IDF, entity/proper-noun first).
    No LLM — fully deterministic, instant, auditable. Queries -> queries.jsonl.
  * Gold = that target drawer's id. Known by CONSTRUCTION, not defined by any
    ranker. Vector and BM25 contribute independent signal; exact vector search
    is no longer trivially 1.0; hybrid can legitimately win or lose.
  * Metric = hit@k: is the target drawer in the strategy's top-k?
    (One gold per query, so hit@k == recall@k == fraction of queries that found it.)

Strategies: exact vector, turbovec, BM25, turbovec+BM25 (RRF); with --chroma also
chromadb alone + chroma+BM25.

READ THE RESULT THIS WAY:
  Template queries are keyword-derived, so they tilt toward BM25 (lexical match
  by construction). This is the BM25-FAVORABLE end of the query spectrum.
    - hybrid NOT beating turbovec-alone even here  -> strong negative for hybrid.
    - hybrid winning                               -> expected; confirm later on
                                                      harder paraphrased queries.
  Absolute hit@k is also pessimistic (a short query may fit several near-dup
  drawers; only the source is credited) but FAIR across strategies — read the
  cross-strategy comparison, not the absolutes.

Usage:
    scp turbovec-target.py corpus.txt spark:~/
    ssh spark 'source ~/.venv/bin/activate && python turbovec-target.py --corpus corpus.txt --chroma'
"""

import argparse
import json
import math
import os
import re
import sys
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from turbovec import IdMapIndex

EMBED = "http://localhost:8000/v1/embeddings"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
CHAT = "http://localhost:8001/v1"
DIM = 768
_TOK = re.compile(r"[a-z0-9]+")
_PROPER = re.compile(r"\b([A-Z][a-z]{2,})")  # entity-ish: Capitalized words

# Renderer prompt: LLM drafts a natural question; the TARGET drawer is fixed by
# construction, so the LLM makes no scope/attribution decision (pipeline rule).
GEN_PROMPT = (
    "Read the passage and write ONE short, natural question (under 15 words) that "
    "a player or DM would actually ask, which THIS passage answers. Paraphrase — "
    "do NOT copy phrases from the passage. Output only the question.\n\n"
    "Passage:\n{doc}\n\nQuestion:"
)

STOP = set(
    "the a an and or but of to in on at by for with from as is are was were be been "
    "being it its this that these those he she his her him they them their there here "
    "you your we our us i me my who what when where which while into out up down over "
    "under all any some no not so if then than too very can will would could should may "
    "might must do does did has have had not no yes one two three more most also them "
    "about after before between during without within against toward upon".split()
)


def embed_all(texts, batch=64, label="embedding", prefix=""):
    # nomic-embed-text-v1.5 is trained with task prefixes; asymmetric retrieval
    # needs "search_document: " on corpus and "search_query: " on queries.
    out = []
    for i in range(0, len(texts), batch):
        chunk = [prefix + t for t in texts[i : i + batch]]
        r = requests.post(EMBED, json={"model": EMBED_MODEL, "input": chunk}, timeout=120)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        out.extend(d["embedding"] for d in data)
        print(f"  {label} {min(i + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    v = np.asarray(out, dtype=np.float32)
    assert v.shape[1] == DIM, f"expected {DIM}-d, got {v.shape[1]}"
    return v


def l2_normalize(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


def load_corpus(path, limit):
    seen, docs = set(), []
    for line in open(path, encoding="utf-8"):
        t = line.strip()
        if t and t not in seen:
            seen.add(t)
            docs.append(t)
        if limit and len(docs) >= limit:
            break
    if not docs:
        sys.exit(f"corpus {path!r} had no usable lines")
    return docs


def tokenize(s):
    return _TOK.findall(s.lower())


def template_query(doc, idf, n_terms=3):
    """Short keyword query from the drawer's most distinctive terms.

    Score candidate tokens by corpus IDF (rare == distinctive == points at THIS
    drawer), boosting entity/proper-noun tokens. Emit terms in document order so
    the query reads naturally ('Glasstaff Redbrand hideout')."""
    propers = {m.lower() for m in _PROPER.findall(doc)}
    toks = tokenize(doc)
    order = {}
    for i, t in enumerate(toks):
        order.setdefault(t, i)
    cand = []
    seen = set()
    for t in toks:
        if t in seen or t in STOP or len(t) < 3 or t.isdigit():
            continue
        seen.add(t)
        # distinctiveness; +entity boost so NPC/place names lead
        score = idf.get(t, 0.0) + (2.0 if t in propers else 0.0)
        cand.append((t, score))
    cand.sort(key=lambda x: -x[1])
    picks = [t for t, _ in cand[:n_terms]]
    picks.sort(key=lambda t: order.get(t, 0))
    return " ".join(picks)


def chat_model_id():
    r = requests.get(f"{CHAT}/models", timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def gen_query(model, doc):
    """One natural question for one drawer. Returns cleaned text or None."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": GEN_PROMPT.format(doc=doc[:2000])}],
        "temperature": 0.7,
        "max_tokens": 64,
    }
    try:
        r = requests.post(f"{CHAT}/chat/completions", json=body, timeout=120)
        r.raise_for_status()
        q = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None
    q = q.splitlines()[-1].strip() if q else q
    q = re.sub(r"^(question:|q:)\s*", "", q, flags=re.I).strip().strip('"').strip()
    return q or None


def build_queries_llm(model, docs, tids, workers=8):
    results = [None] * len(tids)

    def work(idx):
        return idx, gen_query(model, docs[tids[idx]])

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, q in ex.map(work, range(len(tids))):
            results[idx] = (int(tids[idx]), q)
            done += 1
            print(f"  generated {done}/{len(tids)}", end="\r", flush=True)
    print()
    return [(t, q) for (t, q) in results if q]


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.dl = [len(d) for d in docs_tokens]
        self.avgdl = sum(self.dl) / self.N
        self.tf = [Counter(d) for d in docs_tokens]
        df = defaultdict(int)
        self.inv = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                df[t] += 1
                self.inv[t].append(i)
        self.idf = {t: math.log(1 + (self.N - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def rank(self, query_tokens, topn):
        scores = defaultdict(float)
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.inv[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:topn]]


def rrf(rank_lists, topn, k0=60):
    s = defaultdict(float)
    for rl in rank_lists:
        for r, did in enumerate(rl):
            s[did] += 1.0 / (k0 + r + 1)
    return [i for i, _ in sorted(s.items(), key=lambda x: -x[1])[:topn]]


def exact_cands(C, Q, pool):
    sims = Q @ C.T
    out = []
    for row in sims:
        part = np.argpartition(-row, pool)[:pool]
        out.append(part[np.argsort(-row[part])].tolist())
    return out


def turbovec_cands(C, Q, pool, bits=4):
    idx = IdMapIndex(dim=DIM, bit_width=bits)
    idx.add_with_ids(C, np.arange(C.shape[0], dtype=np.uint64))
    out = []
    for q in Q:
        _, ids = idx.search(q.reshape(1, -1).astype(np.float32), k=pool)
        out.append([int(i) for i in ids[0].tolist()])
    return out


def chroma_cands(C, Q, pool):
    import chromadb

    client = chromadb.EphemeralClient()
    coll = client.create_collection("target_ab", metadata={"hnsw:space": "cosine"})
    ids = [str(i) for i in range(C.shape[0])]
    for i in range(0, C.shape[0], 5000):
        sl = slice(i, i + 5000)
        coll.add(ids=ids[sl], embeddings=C[sl].tolist())
    out = []
    for q in Q:
        res = coll.query(query_embeddings=[q.tolist()], n_results=pool)
        out.append([int(x) for x in res["ids"][0]])
    return out


def hit_at_k(cands, targets, k):
    return sum(1 for c, t in zip(cands, targets) if t in c[:k]) / len(cands)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--num-queries", type=int, default=300)
    ap.add_argument("--k", default="1,10,100")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--gen", choices=["template", "llm"], default="template",
                    help="template = deterministic keywords (BM25-favorable); "
                         "llm = natural paraphrased questions (balanced regime)")
    ap.add_argument("--terms", type=int, default=3, help="keywords per template query")
    ap.add_argument("--workers", type=int, default=8, help="concurrent chat requests (llm mode)")
    ap.add_argument("--chroma", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--queries-file", default="queries.jsonl")
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()

    docs = load_corpus(args.corpus, args.n)
    N = len(docs)
    ks = sorted({int(x) for x in args.k.split(",") if x.strip()})
    ks = [k for k in ks if k <= min(N, args.pool)]
    if not ks:
        sys.exit("corpus/pool too small for requested k")

    print(f"corpus: {N} docs | k: {ks} | bits: {args.bits} | pool: {args.pool} | gen: {args.gen}\n")
    if args.gen == "template":
        print("READ: template queries are keyword-derived -> tilt toward BM25 (the")
        print("      BM25-favorable end). A hybrid loss here is decisive.")
    else:
        print("READ: llm queries are natural paraphrased questions (balanced regime)")
        print("      where both modalities contribute -- the regime that exercises")
        print("      turbovec's deficit.")
    print("      Absolute hit@k pessimistic (multi-valid targets) but FAIR across")
    print("      strategies. Compare turbovec+BM25 vs turbovec-alone vs chromadb-alone.\n")

    # BM25 needed both for queries (idf) and as a ranker.
    print("building BM25 index...")
    bm = BM25([tokenize(d) for d in docs])

    # --- queries: load or build deterministically ---
    if os.path.exists(args.queries_file) and not args.regen:
        pairs = [json.loads(l) for l in open(args.queries_file)]
        pairs = [(p["tid"], p["q"]) for p in pairs]
        print(f"loaded {len(pairs)} queries from {args.queries_file} (use --regen to rebuild)\n")
    else:
        rnd = random.Random(args.seed)
        tids = rnd.sample(range(N), min(args.num_queries, N))
        if args.gen == "llm":
            model = chat_model_id()
            print(f"generating {len(tids)} natural questions via vllm-chat: {model}")
            print("  (renderer — target is fixed by construction; eyeball the sample)")
            pairs = build_queries_llm(model, docs, tids, workers=args.workers)
        else:
            pairs = [(int(tid), template_query(docs[tid], bm.idf, n_terms=args.terms))
                     for tid in tids]
            pairs = [(t, q) for t, q in pairs if q]
        with open(args.queries_file, "w") as fh:
            for tid, q in pairs:
                fh.write(json.dumps({"tid": tid, "q": q}) + "\n")
        print(f"built {len(pairs)} {args.gen} queries -> {args.queries_file}\n")

    if not pairs:
        sys.exit("no usable queries")

    targets = [t for t, _ in pairs]
    qtexts = [q for _, q in pairs]

    print("sample (query -> target drawer):")
    for tid, q in pairs[:8]:
        print(f"  Q: {q}")
        print(f"     -> [{tid}] {docs[tid][:100]}...")
    print()

    print("embedding corpus (search_document: prefix)...")
    C = l2_normalize(embed_all(docs, args.batch, label="corpus", prefix="search_document: "))
    print("embedding queries (search_query: prefix)...")
    Q = l2_normalize(embed_all(qtexts, args.batch, label="query", prefix="search_query: "))

    print("exact vector...")
    ex = exact_cands(C, Q, args.pool)
    print("turbovec...")
    tv = turbovec_cands(C, Q, args.pool, bits=args.bits)
    print("BM25...")
    bmc = [bm.rank(tokenize(q), args.pool) for q in qtexts]
    print("turbovec+BM25 (RRF)...")
    tv_bm = [rrf([tv[i], bmc[i]], args.pool, k0=args.rrf_k) for i in range(len(pairs))]

    strategies = [
        ("exact vector", ex),
        (f"turbovec {args.bits}-bit", tv),
        ("BM25 alone", bmc),
        ("turbovec+BM25", tv_bm),
    ]

    if args.chroma:
        print("chroma...")
        try:
            ch = chroma_cands(C, Q, args.pool)
            ch_bm = [rrf([ch[i], bmc[i]], args.pool, k0=args.rrf_k) for i in range(len(pairs))]
            strategies += [("chromadb alone", ch), ("chroma+BM25", ch_bm)]
        except ImportError:
            print("  chromadb not installed — skipping\n")

    print("\n" + "=" * 56)
    print(f"hit@k over {len(pairs)} {args.gen} queries (gold = source drawer)\n")
    rk = "   ".join(f"@{k:<5}" for k in ks)
    print(f"{'strategy':<18}{rk}")
    print("-" * 56)
    for name, cands in strategies:
        cells = "   ".join(f"{hit_at_k(cands, targets, k):.3f}" for k in ks)
        print(f"{name:<18}{cells}")
    print("=" * 56)
    print("\nDECISION: turbovec+BM25 vs turbovec-alone vs chromadb-alone.")
    print("Queries tilt to BM25, so a hybrid loss here is decisive; a win needs")
    print("confirmation on harder (paraphrased) queries before trusting it.")


if __name__ == "__main__":
    main()
