#!/usr/bin/env python3
"""Local parity: turbovec vs chroma mempalace backends, same vectors, same queries.

This runs entirely on the LOCAL machine (no Spark, no GPU): embeddings come from
mempalace's default local ONNX MiniLM (384-d), and both backends are driven
through the real `mempalace.backends` BaseBackend interface and ranked with the
real `searcher._hybrid_rank` (BM25 + cosine). The question it answers is the one
the eval (PR #5) raised at the index level, now at the *backend* level: is the
turbovec backend quality-indistinguishable from the chroma backend end-to-end?

Method:
  * Embed every corpus doc once → feed the SAME float32 vectors to both backends
    (precomputed embeddings, so the only variable is the index/quantization).
  * Each doc gets a wing/room so we can also parity-check a filtered query.
  * Queries are a short held-out word-span of a random doc; gold = that doc id
    (structural gold, not vector-defined — so no circularity). hit@k = gold in
    the top-k after hybrid ranking.
  * Report hit@k for vector-only and hybrid, for each backend, plus inter-backend
    agreement (top-1 match rate, mean Jaccard@10).

The embedder is whatever mempalace is configured to use. To force the fully
local ONNX MiniLM (no network, no Spark) regardless of config, set
MEMPALACE_EMBEDDING_PROVIDER=onnx — the embedder line in the output reports
what was actually used.

Usage (inside the venv that has mempalace + turbovec):
    MEMPALACE_EMBEDDING_PROVIDER=onnx \\
        python turbovec-backend-parity.py --corpus corpus.txt --n 0 --queries 300
"""

import argparse
import os
import random
import re
import shutil
import sys
import tempfile
import time

import numpy as np

from mempalace.backends import get_backend
from mempalace.embedding import get_embedding_function
from mempalace.searcher import _hybrid_rank

_WORD = re.compile(r"\S+")


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


def embed_all(ef, texts, batch=256):
    out = []
    for i in range(0, len(texts), batch):
        out.extend(ef(texts[i : i + batch]))
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    return np.asarray(out, dtype=np.float32)


def short_query(doc, start=4, n=12):
    """A held-out word-span — natural text, but short enough to be a real
    retrieval problem (full-doc-as-query would trivially self-match)."""
    words = _WORD.findall(doc)
    if len(words) <= n:
        return doc
    s = min(start, max(0, len(words) - n))
    return " ".join(words[s : s + n])


def build_collection(backend_name, palace_dir, docs, ids, metas, vecs):
    """Returns (backend, col, build_seconds) — build_seconds times only the
    add() loop (index construction), not embedding."""
    backend = get_backend(backend_name)
    col = backend.get_collection(palace_dir, collection_name="mempalace_drawers", create=True)
    B = 1000
    t0 = time.perf_counter()
    for i in range(0, len(docs), B):
        sl = slice(i, i + B)
        col.add(
            documents=docs[sl],
            ids=ids[sl],
            metadatas=metas[sl],
            embeddings=[v.tolist() for v in vecs[sl]],
        )
    return backend, col, time.perf_counter() - t0


def ranked_ids(col, qvec, query_text, pool, k, where=None):
    """Return (vector_only_topk_ids, hybrid_topk_ids, query_seconds).

    query_seconds times only the backend ``col.query`` call — the ANN search
    plus (for turbovec) the exact-cosine re-rank. The shared ``_hybrid_rank``
    BM25 pass runs above the backend and is excluded so the number is
    backend-attributable."""
    t0 = time.perf_counter()
    res = col.query(
        query_embeddings=[qvec.tolist()],
        n_results=pool,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    q_s = time.perf_counter() - t0
    rids = res.ids[0]
    rdocs = res.documents[0]
    rdists = res.distances[0]
    vector_only = rids[:k]
    hits = [{"id": rid, "text": d, "distance": dist}
            for rid, d, dist in zip(rids, rdocs, rdists)]
    _hybrid_rank(hits, query_text)
    hybrid = [h["id"] for h in hits[:k]]
    return vector_only, hybrid, q_s


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 * 1024)


def hit_at_k(ranked, gold, ks):
    return {k: (1.0 if gold in ranked[:k] else 0.0) for k in ks}


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--n", type=int, default=4000, help="docs to index (0 = all)")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--pool", type=int, default=50, help="candidates per query before re-rank")
    ap.add_argument("--k", default="1,5,10")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ks = sorted({int(x) for x in args.k.split(",") if x.strip()})
    docs = load_corpus(args.corpus, args.n)
    N = len(docs)
    ids = [f"d{i}" for i in range(N)]
    metas = [{"wing": f"w{i % 8}", "room": f"r{i % 32}"} for i in range(N)]

    from mempalace.config import MempalaceConfig
    cfg = MempalaceConfig()
    prov = cfg.embedding_provider
    model = getattr(cfg, "embedding_model", "?")
    endpoint = getattr(cfg, "embedding_endpoint", "")
    print(f"corpus: {N} docs | queries: {min(args.queries, N)} | pool: {args.pool} | k: {ks}")
    print(f"embedder: provider={prov} model={model} endpoint={endpoint or '(local onnx)'}")
    print("(both backends get the SAME vectors — the only variable is the index)\n")

    ef = get_embedding_function()
    print("embedding corpus (once; fed identically to both backends)...")
    vecs = embed_all(ef, docs)
    dim = vecs.shape[1]
    print(f"  dim={dim}\n")

    rnd = random.Random(args.seed)
    q_idx = rnd.sample(range(N), min(args.queries, N))
    q_texts = [short_query(docs[i]) for i in q_idx]
    print("embedding queries...")
    q_vecs = embed_all(ef, q_texts)

    tmp = tempfile.mkdtemp(prefix="tv-parity-")
    try:
        results = {}      # backend -> {"vec": {k:hits}, "hyb": {k:hits}}
        topk_ids = {}     # backend -> list of hybrid top-k id lists (for agreement)
        filt = {}         # backend -> filtered-query spot check passed?
        perf = {}         # backend -> {build_s, p50_ms, p95_ms, mean_ms, size_mb}
        target = next(i for i in range(N) if metas[i]["room"] == "r0")
        for name in ("chroma", "turbovec"):
            print(f"\n=== backend: {name} ===")
            palace_dir = os.path.join(tmp, name)
            backend, col, build_s = build_collection(name, palace_dir, docs, ids, metas, vecs)
            print(f"  indexed {col.count()} docs in {build_s:.2f}s")
            vec_acc = {k: 0.0 for k in ks}
            hyb_acc = {k: 0.0 for k in ks}
            per_query_hybrid = []
            q_times = []
            for qi, qtext, qvec in zip(q_idx, q_texts, q_vecs):
                gold = ids[qi]
                vonly, hyb, q_s = ranked_ids(col, qvec, qtext, args.pool, max(ks))
                q_times.append(q_s * 1000.0)  # ms
                per_query_hybrid.append(hyb)
                for k, v in hit_at_k(vonly, gold, ks).items():
                    vec_acc[k] += v
                for k, v in hit_at_k(hyb, gold, ks).items():
                    hyb_acc[k] += v
            nq = len(q_idx)
            results[name] = {
                "vec": {k: vec_acc[k] / nq for k in ks},
                "hyb": {k: hyb_acc[k] / nq for k in ks},
            }
            topk_ids[name] = per_query_hybrid

            # filtered-query spot check on the same open collection: every hit
            # must be in room r0, and the on-target doc must come back.
            _, hyb, _ = ranked_ids(col, vecs[target], short_query(docs[target]),
                                   args.pool, 5, where={"room": "r0"})
            got = col.get(ids=hyb, include=["metadatas"])
            rooms = [m.get("room") for m in got.metadatas]
            filt[name] = bool(hyb) and all(r == "r0" for r in rooms) and ids[target] in hyb

            backend.close()  # flushes turbovec .tvim → measure size after
            perf[name] = {
                "build_s": build_s,
                "p50_ms": float(np.percentile(q_times, 50)),
                "p95_ms": float(np.percentile(q_times, 95)),
                "mean_ms": float(np.mean(q_times)),
                "size_mb": dir_size_mb(palace_dir),
            }

        # ---- quality report ----
        print("\n" + "=" * 64)
        header = "  ".join(f"v@{k:<3}" for k in ks) + "   " + "  ".join(f"h@{k:<3}" for k in ks)
        print(f"{'backend':<10}{header}")
        print("-" * 64)
        for name in ("chroma", "turbovec"):
            r = results[name]
            cells = "  ".join(f"{r['vec'][k]:.3f}" for k in ks)
            cells += "   " + "  ".join(f"{r['hyb'][k]:.3f}" for k in ks)
            print(f"{name:<10}{cells}")
        print("=" * 64)

        # inter-backend agreement on hybrid top-k
        top1_match = np.mean([
            (a[:1] == b[:1]) for a, b in zip(topk_ids["chroma"], topk_ids["turbovec"])
        ])
        jacc = np.mean([
            jaccard(a, b) for a, b in zip(topk_ids["chroma"], topk_ids["turbovec"])
        ])
        print(f"\ninter-backend agreement (hybrid): top-1 match {top1_match:.1%}, "
              f"mean Jaccard@{max(ks)} {jacc:.3f}")
        print(f"filtered-query (room=r0) honored + on-target: "
              f"chroma={filt['chroma']}, turbovec={filt['turbovec']}")
        print("v@k = vector-only hit@k; h@k = BM25+vector hybrid hit@k.")

        # ---- performance report (measured, this machine, same vectors) ----
        print("\n" + "=" * 64)
        print(f"{'backend':<10}{'build s':>9}{'q p50 ms':>11}{'q p95 ms':>11}"
              f"{'q mean ms':>11}{'size MB':>10}")
        print("-" * 64)
        for name in ("chroma", "turbovec"):
            p = perf[name]
            print(f"{name:<10}{p['build_s']:>9.2f}{p['p50_ms']:>11.2f}{p['p95_ms']:>11.2f}"
                  f"{p['mean_ms']:>11.2f}{p['size_mb']:>10.1f}")
        print("=" * 64)

        def _ratio(metric, lower_better=True):
            c, t = perf["chroma"][metric], perf["turbovec"][metric]
            if t == 0 or c == 0:
                return "n/a"
            r = c / t if lower_better else t / c
            return f"{r:.1f}x"
        print(f"\nturbovec vs chroma: build {_ratio('build_s')} faster, "
              f"q p50 {_ratio('p50_ms')} faster, q p95 {_ratio('p95_ms')} faster, "
              f"size {perf['turbovec']['size_mb']/perf['chroma']['size_mb']:.2f}x "
              f"(>1 = turbovec larger; it stores float32 for exact re-rank).")
        print("q time = backend col.query() only (ANN + turbovec re-rank); "
              "the shared BM25 pass is excluded.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
