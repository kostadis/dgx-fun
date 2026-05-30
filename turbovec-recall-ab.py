#!/usr/bin/env python3
"""turbovec recall A/B — the test that decides adoption against mempalace's bar.

Runs on the Spark (ssh spark), vllm-embed up on :8000. The smoke test proved
turbovec *runs*; this measures whether 4-bit TurboQuant keeps enough recall to
matter for a verbatim-100%-recall memory system.

The method (see turbovec-smoke.py for the install/correctness gate this builds on):

  1. Load a real corpus (one doc per line). Toy data lies; pass real text.
  2. Embed everything ONCE through vllm-embed -> an (N, 768) float32 matrix.
     Both indexes get the SAME vectors, so we measure the index, not the embedder.
  3. L2-normalize. Now exact dot product == cosine, and turbovec's quantized
     score approximates that same cosine. Apples to apples.
  4. Ground truth: exact brute-force kNN in numpy. Slow, perfect, the answer key.
  5. Candidates: turbovec 4-bit, and (optionally) ChromaDB HNSW -- the backend
     turbovec would replace. Chroma's ANN isn't 1.0 either; comparing the two is
     fairer than holding turbovec against a perfection nobody ships.
  6. recall@k = |candidate_topk ∩ exact_topk| / k, averaged over held-out queries.

What the number decides:
  ~0.95+      -> 4-bit loses a little; next question is whether mempalace's
                 hybrid BM25+vector layer recovers the misses. Worth a backend.
  << chroma   -> you're paying recall for an 8x-smaller / faster CPU index.
                 That's the calibration lesson, in a hard number.
  < ~0.85     -> disqualifying for a 100%-recall system. Learned cheaply.

Usage:
    ssh spark && source ~/.venv/bin/activate
    # real corpus is the whole point -- scp some real text over first:
    #   scp ~/src/campaigns-test/**/*.md spark:~/corpus/   (or any prose)
    python turbovec-recall-ab.py --corpus corpus.txt
    python turbovec-recall-ab.py --corpus corpus.txt --chroma --num-queries 500
    python turbovec-recall-ab.py            # no --corpus: tiny built-in demo, wiring only
"""

import argparse
import os
import sys
import time
import random
import tempfile

import numpy as np
import requests
from turbovec import IdMapIndex

EMBED = "http://localhost:8000/v1/embeddings"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # exact id from /v1/models
DIM = 768

# Smoke-grade fallback ONLY. A few dozen sentences makes recall@10 trivially
# ~1.0 and decides nothing -- it exists so you can prove the harness is wired
# before you bother copying a real corpus over. The WARNING below is load-bearing.
_DEMO_CORPUS = [
    "The DGX Spark runs Qwen3-Next 80B A3B on the Blackwell GPU.",
    "turbovec is a CPU vector index built on TurboQuant 4-bit quantization.",
    "nomic-embed-text-v1.5 produces 768-dimensional sentence embeddings.",
    "ChromaDB uses HNSW for approximate nearest-neighbor search.",
    "Cacio e pepe is a Roman pasta of pecorino, pepper, and starchy water.",
    "The Grace CPU and the GPU share one pool of LPDDR5X memory.",
    "vLLM serves an OpenAI-compatible API for chat and embeddings.",
    "BM25 ranks documents by sparse lexical term-frequency overlap.",
    "mempalace stores user text verbatim and targets one hundred percent recall.",
    "Speculative decoding pairs a small draft model with a large target model.",
    "L2 normalization turns a dot product into a cosine similarity.",
    "HNSW trades a little recall for a large speedup over brute force.",
    "The method of loci places memories in the rooms of an imagined building.",
    "Carbonara adds egg and guanciale where cacio e pepe stays bare.",
    "Grace Blackwell is an ARM aarch64 platform, not x86.",
    "Recall at k measures how many true neighbors an index actually returns.",
    "Ollama runs quantized GGUF models on local hardware.",
    "Prefill is compute-bound; decode is memory-bandwidth-bound.",
    "A reranker reorders a candidate set after first-stage retrieval.",
    "The Spark's idle ARM cores can host a vector index beside the GPU's LLM.",
]


def embed_all(texts, batch=64):
    """Embed every text through vllm-embed in batches, return (N, 768) float32."""
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        r = requests.post(EMBED, json={"model": EMBED_MODEL, "input": chunk}, timeout=120)
        r.raise_for_status()
        data = r.json()["data"]
        # vLLM preserves input order in `data`, but key on `index` to be safe.
        data = sorted(data, key=lambda d: d["index"])
        out.extend(d["embedding"] for d in data)
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    v = np.asarray(out, dtype=np.float32)
    assert v.shape[1] == DIM, f"expected {DIM}-d, got {v.shape[1]}"
    return v


def l2_normalize(m):
    """Row-wise unit norm so dot == cosine. Guard the zero-vector edge case."""
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


def load_corpus(path, limit):
    """One non-blank doc per line. Dedup exact repeats (they skew recall)."""
    seen, docs = set(), []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            t = line.strip()
            if t and t not in seen:
                seen.add(t)
                docs.append(t)
            if limit and len(docs) >= limit:
                break
    if not docs:
        sys.exit(f"corpus {path!r} had no usable lines")
    return docs


def exact_topn(C, q_idx, topn):
    """Brute-force ground truth: for each query row, the topn nearest corpus ids,
    excluding the query's own id. C is L2-normalized, so C@q == cosine."""
    res = []
    Q = C[q_idx]                       # (nq, dim)
    sims = Q @ C.T                     # (nq, N) cosine — full matrix; batch if huge
    for row, self_id in zip(sims, q_idx):
        row = row.copy()
        row[self_id] = -np.inf         # drop self
        # argpartition for topn, then sort just those by score desc
        part = np.argpartition(-row, topn)[:topn]
        order = part[np.argsort(-row[part])]
        res.append(order.tolist())
    return res


def turbovec_topn(C, q_idx, topn, bits=4):
    """Quantized candidate at `bits` bit-width. Returns
    (per-query topn ids excl. self, build_s, latencies, on-disk size)."""
    t0 = time.perf_counter()
    idx = IdMapIndex(dim=DIM, bit_width=bits)
    idx.add_with_ids(C, np.arange(C.shape[0], dtype=np.uint64))
    build_s = time.perf_counter() - t0

    queries = np.ascontiguousarray(C[q_idx], dtype=np.float32)
    # Per-query latency is the honest metric for an interactive memory lookup,
    # so time them one at a time rather than reporting batched throughput.
    lat, ids_out = [], []
    for q, self_id in zip(queries, q_idx):
        q2 = q.reshape(1, -1)
        s = time.perf_counter()
        _, ids = idx.search(q2, k=topn + 1)   # +1 to survive self-drop
        lat.append(time.perf_counter() - s)
        row = [int(i) for i in ids[0].tolist() if int(i) != self_id][:topn]
        ids_out.append(row)

    # Index footprint on disk: the headline number for a quantized index.
    size = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tv", delete=False) as tf:
            tmp = tf.name
        idx.write(tmp)
        size = os.path.getsize(tmp)
        os.unlink(tmp)
    except Exception:
        pass
    return ids_out, build_s, lat, size


def chroma_topn(C, q_idx, topn):
    """Optional ChromaDB HNSW comparison, fed the SAME precomputed vectors."""
    import chromadb

    t0 = time.perf_counter()
    client = chromadb.EphemeralClient()
    coll = client.create_collection("recall_ab", metadata={"hnsw:space": "cosine"})
    ids = [str(i) for i in range(C.shape[0])]
    # Chroma chokes on very large single adds; chunk it.
    for i in range(0, C.shape[0], 5000):
        sl = slice(i, i + 5000)
        coll.add(ids=ids[sl], embeddings=C[sl].tolist())
    build_s = time.perf_counter() - t0

    lat, ids_out = [], []
    for qi in q_idx:
        s = time.perf_counter()
        res = coll.query(query_embeddings=[C[qi].tolist()], n_results=topn + 1)
        lat.append(time.perf_counter() - s)
        row = [int(x) for x in res["ids"][0] if int(x) != qi][:topn]
        ids_out.append(row)
    return ids_out, build_s, lat


def recall_at_k(approx, exact, k):
    """Mean over queries of |approx_topk ∩ exact_topk| / k."""
    tot = 0.0
    for a, e in zip(approx, exact):
        tot += len(set(a[:k]) & set(e[:k])) / k
    return tot / len(approx)


def pctl(xs, p):
    return float(np.percentile(np.asarray(xs) * 1000.0, p))  # ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="text file, one doc per line (real prose — toy data lies)")
    ap.add_argument("--n", type=int, default=0, help="cap corpus size (0 = all)")
    ap.add_argument("--num-queries", type=int, default=300, help="held-out queries sampled from corpus")
    ap.add_argument("--k", default="1,10,100", help="comma list of k for recall@k")
    ap.add_argument("--bits", default="4", help="comma list of turbovec bit-widths to sweep, e.g. 1,2,4,8")
    ap.add_argument("--chroma", action="store_true", help="also benchmark ChromaDB HNSW")
    ap.add_argument("--seed", type=int, default=0, help="query-sample seed (reproducible)")
    ap.add_argument("--batch", type=int, default=64, help="embedding request batch size")
    args = ap.parse_args()

    if args.corpus:
        docs = load_corpus(args.corpus, args.n)
    else:
        docs = list(_DEMO_CORPUS)
        print("WARNING: no --corpus given; using the tiny built-in demo set.")
        print("         This validates wiring ONLY. Recall on ~20 docs is meaningless")
        print("         for a real decision — pass --corpus with real prose.\n")

    ks = sorted({int(x) for x in args.k.split(",") if x.strip()})
    N = len(docs)
    # k can't exceed the pool minus the held-out self.
    ks = [k for k in ks if k <= N - 1]
    if not ks:
        sys.exit(f"corpus too small ({N} docs) for any requested k")
    topn = max(ks)

    print(f"corpus: {N} docs | queries: {min(args.num_queries, N)} | k: {ks}\n")

    print("embedding corpus through vllm-embed...")
    C = l2_normalize(embed_all(docs, args.batch))

    rnd = random.Random(args.seed)
    q_idx = np.array(rnd.sample(range(N), min(args.num_queries, N)), dtype=np.int64)

    print("exact kNN ground truth (brute force)...")
    g0 = time.perf_counter()
    exact = exact_topn(C, q_idx, topn)
    exact_s = time.perf_counter() - g0

    rows = []
    rows.append(("exact kNN", {k: 1.0 for k in ks}, exact_s / len(q_idx), exact_s, N * DIM * 4))

    bit_widths = [int(b) for b in args.bits.split(",") if b.strip()]
    for bits in bit_widths:
        print(f"turbovec {bits}-bit...")
        try:
            tv_ids, tv_build, tv_lat, tv_size = turbovec_topn(C, q_idx, topn, bits=bits)
        except Exception as e:
            # Unsupported bit-width should report, not abort the whole sweep.
            print(f"  {bits}-bit unsupported/failed: {type(e).__name__}: {e}\n")
            continue
        rows.append((
            f"turbovec {bits}-bit",
            {k: recall_at_k(tv_ids, exact, k) for k in ks},
            None, tv_build, tv_size, tv_lat,
        ))

    if args.chroma:
        print("chromadb HNSW...")
        try:
            ch_ids, ch_build, ch_lat = chroma_topn(C, q_idx, topn)
            rows.append((
                "chromadb HNSW",
                {k: recall_at_k(ch_ids, exact, k) for k in ks},
                None, ch_build, None, ch_lat,
            ))
        except ImportError:
            print("  chromadb not installed in this venv — skipping (pip install chromadb)\n")

    # ---- report ----
    print("\n" + "=" * 72)
    rk = "  ".join(f"r@{k:<4}" for k in ks)
    print(f"{'index':<16}{rk}   p50 ms   p95 ms   build s   size")
    print("-" * 72)
    for row in rows:
        name, rec = row[0], row[1]
        rcells = "  ".join(f"{rec[k]:.3f}" for k in ks)
        lat = row[5] if len(row) > 5 else None
        p50 = f"{pctl(lat, 50):7.2f}" if lat else (f"{row[2] * 1000:7.2f}" if row[2] else "      -")
        p95 = f"{pctl(lat, 95):7.2f}" if lat else "      -"
        build = f"{row[3]:7.2f}" if row[3] is not None else "      -"
        size = f"{row[4] / 1e6:6.1f}MB" if row[4] else "     -"
        print(f"{name:<16}{rcells}   {p50}  {p95}  {build}  {size}")
    print("=" * 72)
    print("\nrecall is vs exact kNN. turbovec/chroma size 'size' = on-disk index;")
    print("exact 'size' = raw float32 matrix (the thing quantization shrinks).")
    print("scores from turbovec are higher-is-better and unnormalized — ranking only.")


if __name__ == "__main__":
    main()
