#!/usr/bin/env python3
"""embed-ab.py — end-to-end retrieval A/B: nomic-embed-text vs qwen3-embedding:0.6b.

Method: HELD-OUT SPAN (no LLM in the loop, deterministic, fair to both).
  For each sampled doc, pull one sentence out as the QUERY; index the doc
  with that sentence REMOVED. A query "hits" if its parent doc ranks top-k
  among all sampled docs. Same docs + same query spans for both models, so
  we measure the EMBEDDER, end-to-end (hit@k / MRR), not index recall.

Fairness: each model gets its NATIVE prefixes —
  nomic : doc  -> "search_document: <text>",  query -> "search_query: <text>"
  qwen3 : doc  -> "<text>",                    query -> "Instruct: ...\nQuery: <text>"
Skipping nomic's task prefixes would handicap it; skipping qwen3's query
instruction would handicap it. Both applied. (Assumes Ollama /api/embed does
NOT auto-prefix — it embeds raw input. Verified separately.)

Usage:
  python3 embed-ab.py --corpus corpus.txt --n 400 --k 1 5 10 --seed 7
Hits Ollama on spark1 (192.168.1.147:11434) by default.
"""
import argparse, re, sys, time, random
import numpy as np
import requests

OLLAMA = "http://192.168.1.147:11434/api/embed"
SENT = re.compile(r'(?<=[.!?])\s+')

NOMIC = "nomic-embed-text"
QWEN = "qwen3-embedding:0.6b"
QWEN_INSTR = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "


def load(path, n, seed, min_len=300, min_sents=3):
    docs = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if len(line) < min_len:
                continue
            sents = [s for s in SENT.split(line) if len(s.strip()) >= 40]
            if len(sents) < min_sents:
                continue
            docs.append((line, sents))
    random.Random(seed).shuffle(docs)
    docs = docs[:n]
    queries, indexed = [], []
    for line, sents in docs:
        # held-out span: the middle sentence (avoids title/boilerplate first line)
        qi = len(sents) // 2
        span = sents[qi]
        queries.append(span.strip())
        indexed.append(line.replace(span, " ").strip())
    return queries, indexed


def embed(model, texts, batch=64):
    out = []
    t0 = time.time()
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        r = requests.post(OLLAMA, json={"model": model, "input": chunk}, timeout=600)
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    dt = time.time() - t0
    m = np.asarray(out, dtype=np.float32)
    m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    return m, dt


def evaluate(model, queries, indexed, doc_pre, q_pre, ks):
    dmat, dt_d = embed(model, [doc_pre + t for t in indexed])
    qmat, dt_q = embed(model, [q_pre + t for t in queries])
    n = len(queries)
    sims = qmat @ dmat.T                      # (n_query, n_doc)
    ranks = np.empty(n, dtype=np.int64)
    for i in range(n):
        order = np.argsort(-sims[i])          # best first
        ranks[i] = int(np.where(order == i)[0][0]) + 1   # 1-based rank of parent
    hits = {k: float(np.mean(ranks <= k)) for k in ks}
    mrr = float(np.mean(1.0 / ranks))
    return {"dim": dmat.shape[1], "hits": hits, "mrr": mrr,
            "embed_s": dt_d + dt_q, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--raw", action="store_true",
                    help="No prefixes for either model — matches how mempalace's "
                         "embedding function actually calls them (raw text).")
    args = ap.parse_args()

    queries, indexed = load(args.corpus, args.n, args.seed)
    print(f"corpus={args.corpus}  sampled={len(queries)} docs (held-out middle sentence)  seed={args.seed}\n")

    nomic_dpre, nomic_qpre = ("", "") if args.raw else ("search_document: ", "search_query: ")
    qwen_dpre, qwen_qpre = ("", "") if args.raw else ("", QWEN_INSTR)
    print(f"prefixing: {'RAW (none — production-faithful)' if args.raw else 'native per-model'}\n")
    results = {}
    results[NOMIC] = evaluate(NOMIC, queries, indexed, nomic_dpre, nomic_qpre, args.k)
    results[QWEN] = evaluate(QWEN, queries, indexed, qwen_dpre, qwen_qpre, args.k)

    kcols = "  ".join(f"hit@{k}" for k in args.k)
    print(f"{'model':<26} {'dim':>5}  {kcols}  {'MRR':>6}  {'embed_s':>8}")
    for name, r in results.items():
        hs = "  ".join(f"{r['hits'][k]*100:5.1f}%" for k in args.k)
        print(f"{name:<26} {r['dim']:>5}  {hs}  {r['mrr']:6.3f}  {r['embed_s']:7.1f}s")

    # verdict
    win = lambda k: results[QWEN]['hits'][k] - results[NOMIC]['hits'][k]
    print("\nΔ (qwen3 − nomic):", "  ".join(f"hit@{k} {win(k)*100:+.1f}pt" for k in args.k),
          f"  MRR {results[QWEN]['mrr']-results[NOMIC]['mrr']:+.3f}")


if __name__ == "__main__":
    main()
