#!/usr/bin/env python3
"""turbovec smoke test — run on the Spark (ssh spark), vllm-embed up on :8000.

Answers exactly two questions, nothing else:
  1. Does turbovec install/import on the Spark's aarch64 CPU?
  2. Does it return the obvious neighbor when fed real nomic 768-d vectors?

No mempalace, no sidecar store, no recall benchmark. If this fails, none of
the backend-integration work is worth writing yet. If it prints PASS, the next
step is the recall A/B (turbovec vs ChromaDB on a real corpus), which is the
test that actually decides adoption against mempalace's 100%-recall bar.

Usage:
    ssh spark
    pip install turbovec requests numpy   # watch: wheel vs Rust source build
    python turbovec-smoke.py
"""

import numpy as np
import requests
from turbovec import IdMapIndex

EMBED = "http://localhost:8000/v1/embeddings"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # exact id from /v1/models


def embed(texts):
    r = requests.post(EMBED, json={"model": EMBED_MODEL, "input": texts})
    r.raise_for_status()
    v = np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
    # A wrong dim is silent corruption downstream — fail loud here.
    assert v.shape[1] == 768, f"expected 768-d, got {v.shape[1]}"
    return v


def main():
    docs = [
        "The DGX Spark runs Qwen3-Next 80B on the GPU.",
        "turbovec is a CPU vector index using TurboQuant.",
        "My favorite pasta is cacio e pepe.",
    ]
    vecs = embed(docs)

    idx = IdMapIndex(dim=768, bit_width=4)
    idx.add_with_ids(vecs, np.arange(len(docs), dtype=np.uint64))

    # .search wants a 2-D batch (n_queries, dim) — param is named `queries`.
    q = embed(["what model serves chat completions on the spark?"])
    q = np.ascontiguousarray(q, dtype=np.float32)
    scores, ids = idx.search(q, k=3)

    # Returns are 2-D (n_queries, k); we asked one query, so take row 0.
    top_ids, top_scores = ids[0], scores[0]
    print(list(zip(top_ids.tolist(), top_scores.tolist())))
    # Doc 0 (Spark/Qwen) is the unambiguous answer. If 4-bit quantization
    # can't get even this right, recall on a real corpus is hopeless.
    assert top_ids[0] == 0, f"expected doc 0 (the Spark/Qwen doc) on top, got {top_ids[0]}"
    print("PASS")


if __name__ == "__main__":
    main()
