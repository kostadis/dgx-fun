#!/usr/bin/env python3
"""turbovec + BM25 hybrid — does a lexical backstop recover turbovec's shallow
recall gap vs Chroma? Runs on the Spark, vllm-embed up on :8000.

The recall A/B (turbovec-recall-ab.py) established: at 15k docs turbovec 4-bit
trails Chroma by 3-4 points on r@1/r@10 (the true top-1 vector neighbor is
sometimes ranked #2-#5), while drawing even on r@100. This script asks the only
question that decides adoption for a hybrid system like mempalace: does fusing
BM25 with turbovec's vector candidates float those mis-ranked true neighbors
back to the top?

Method:
  ground truth = exact-vector top-k (same gold standard as the A/B). Every
  strategy is judged on how many of the exact cosine neighbors it returns:
    - turbovec 4-bit alone        (the baseline gap)
    - BM25 alone                  (lexical signal on its own)
    - turbovec + BM25  (RRF)      (the headline: does the gap close?)
    - chromadb alone              (the incumbent target line)        [--chroma]
    - chromadb + BM25  (RRF)      (incumbent's own hybrid ceiling)   [--chroma]

  Fusion is Reciprocal Rank Fusion: score(d) = sum_r 1/(k0 + rank_r(d)). Rank-
  based, so turbovec's unnormalized scores and BM25's scores never have to be
  reconciled on one scale.

HONEST CAVEAT (printed at runtime too): the "query" is a held-out document's
FULL text. BM25 over a whole drawer's vocabulary looks far stronger than BM25
over a real short user query. Treat the lexical lift here as an UPPER BOUND.

Usage:
    scp turbovec-hybrid.py corpus.txt spark:~/
    ssh spark 'source ~/.venv/bin/activate && python turbovec-hybrid.py --corpus corpus.txt --chroma'
"""

import argparse
import math
import re
import sys
import random
from collections import Counter, defaultdict

import numpy as np
import requests
from turbovec import IdMapIndex

EMBED = "http://localhost:8000/v1/embeddings"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DIM = 768
_TOK = re.compile(r"[a-z0-9]+")


def embed_all(texts, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        r = requests.post(EMBED, json={"model": EMBED_MODEL, "input": chunk}, timeout=120)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        out.extend(d["embedding"] for d in data)
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
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


class BM25:
    """Okapi BM25 with an inverted index. Standard k1/b defaults."""

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

    def rank(self, query_tokens, topn, exclude=None):
        scores = defaultdict(float)
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.inv[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        if exclude is not None:
            scores.pop(exclude, None)
        return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:topn]]


def rrf(rank_lists, topn, k0=60, exclude=None):
    s = defaultdict(float)
    for rl in rank_lists:
        for r, did in enumerate(rl):
            s[did] += 1.0 / (k0 + r + 1)
    if exclude is not None:
        s.pop(exclude, None)
    return [i for i, _ in sorted(s.items(), key=lambda x: -x[1])[:topn]]


def exact_topn(C, q_idx, topn):
    res = []
    sims = C[q_idx] @ C.T
    for row, self_id in zip(sims, q_idx):
        row = row.copy()
        row[self_id] = -np.inf
        part = np.argpartition(-row, topn)[:topn]
        res.append(part[np.argsort(-row[part])].tolist())
    return res


def turbovec_cands(C, q_idx, pool, bits=4):
    idx = IdMapIndex(dim=DIM, bit_width=bits)
    idx.add_with_ids(C, np.arange(C.shape[0], dtype=np.uint64))
    out = []
    for qi in q_idx:
        _, ids = idx.search(C[qi].reshape(1, -1).astype(np.float32), k=pool + 1)
        out.append([int(i) for i in ids[0].tolist() if int(i) != qi][:pool])
    return out


def chroma_cands(C, q_idx, pool):
    import chromadb

    client = chromadb.EphemeralClient()
    coll = client.create_collection("hybrid_ab", metadata={"hnsw:space": "cosine"})
    ids = [str(i) for i in range(C.shape[0])]
    for i in range(0, C.shape[0], 5000):
        sl = slice(i, i + 5000)
        coll.add(ids=ids[sl], embeddings=C[sl].tolist())
    out = []
    for qi in q_idx:
        res = coll.query(query_embeddings=[C[qi].tolist()], n_results=pool + 1)
        out.append([int(x) for x in res["ids"][0] if int(x) != qi][:pool])
    return out


def recall_at_k(cands, exact, k):
    tot = 0.0
    for a, e in zip(cands, exact):
        tot += len(set(a[:k]) & set(e[:k])) / k
    return tot / len(cands)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--num-queries", type=int, default=500)
    ap.add_argument("--k", default="1,10,100")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--pool", type=int, default=200, help="candidates per ranker before fusion")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--chroma", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    docs = load_corpus(args.corpus, args.n)
    ks = sorted({int(x) for x in args.k.split(",") if x.strip()})
    N = len(docs)
    ks = [k for k in ks if k <= min(N - 1, args.pool)]
    if not ks:
        sys.exit("corpus/pool too small for requested k")

    print(f"corpus: {N} docs | queries: {min(args.num_queries, N)} | k: {ks} | "
          f"bits: {args.bits} | pool: {args.pool}\n")
    print("CAVEAT: query = held-out document's FULL text. BM25 lift here is an")
    print("        UPPER BOUND vs real short user queries.\n")

    print("embedding corpus...")
    C = l2_normalize(embed_all(docs, args.batch))

    rnd = random.Random(args.seed)
    q_idx = np.array(rnd.sample(range(N), min(args.num_queries, N)), dtype=np.int64)

    print("exact kNN ground truth...")
    exact = exact_topn(C, q_idx, max(ks))

    print("turbovec candidates...")
    tv = turbovec_cands(C, q_idx, args.pool, bits=args.bits)

    print("BM25 index + candidates...")
    bm = BM25([tokenize(d) for d in docs])
    bm_cands = [bm.rank(tokenize(docs[int(qi)]), args.pool, exclude=int(qi)) for qi in q_idx]

    print("fusing turbovec + BM25 (RRF)...")
    tv_bm = [rrf([tv[i], bm_cands[i]], args.pool, k0=args.rrf_k, exclude=int(q_idx[i]))
             for i in range(len(q_idx))]

    strategies = [
        (f"turbovec {args.bits}-bit", tv),
        ("BM25 alone", bm_cands),
        (f"turbovec+BM25", tv_bm),
    ]

    if args.chroma:
        print("chroma candidates...")
        try:
            ch = chroma_cands(C, q_idx, args.pool)
            ch_bm = [rrf([ch[i], bm_cands[i]], args.pool, k0=args.rrf_k, exclude=int(q_idx[i]))
                     for i in range(len(q_idx))]
            strategies.append(("chromadb alone", ch))
            strategies.append(("chroma+BM25", ch_bm))
        except ImportError:
            print("  chromadb not installed — skipping\n")

    # ---- report ----
    print("\n" + "=" * 56)
    rk = "   ".join(f"r@{k:<4}" for k in ks)
    print(f"{'strategy':<18}{rk}")
    print("-" * 56)
    print(f"{'exact (gold)':<18}" + "   ".join(f"{1.0:.3f}" for _ in ks))
    for name, cands in strategies:
        cells = "   ".join(f"{recall_at_k(cands, exact, k):.3f}" for k in ks)
        print(f"{name:<18}{cells}")
    print("=" * 56)
    print("\nrecall vs exact-vector top-k. The decision number: does turbovec+BM25")
    print("r@1 / r@10 close the gap to chromadb alone? (Upper bound — see caveat.)")


if __name__ == "__main__":
    main()
