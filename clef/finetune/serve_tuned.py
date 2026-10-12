"""`vllm-sr-runtime serve`, with Nox's candidate head replaced by a tuned one after the package loads.

    python3 serve_tuned.py TUNED_HEAD.safetensors <the usual serve arguments...>

The released package is loaded and verified exactly as the stock server does (manifest hashes, model
identity, golden check -- all against the released head), then the tuned head's weights are copied into
the loaded head module in place. Backbone, renderer, scheduler and HTTP API are the stock runtime's.
The swap is logged with the file's sha256 so a run can be tied to the head that served it.
"""
import hashlib, logging, sys
from safetensors.torch import load_file
import vllm_sr_runtime.runtime as R
from vllm_sr_runtime.cli import main

TUNED = sys.argv.pop(1)
SHA = hashlib.sha256(open(TUNED, "rb").read()).hexdigest()
_load = R.Runtime.load

def load(self):
    _load(self)                                  # verification + golden check with the released head
    head = self.model.head; p = next(head.parameters())
    sd = {k: v.to(p.device, p.dtype) for k, v in load_file(TUNED).items()}
    head.load_state_dict(sd, strict=True)
    logging.getLogger("vllm_sr_runtime").warning("TUNED HEAD loaded: %s sha256=%s dtype=%s", TUNED, SHA, p.dtype)
    print(f"TUNED HEAD loaded: {TUNED} sha256={SHA} dtype={p.dtype}", flush=True)

R.Runtime.load = load
sys.argv = ["vllm-sr-runtime", "serve"] + sys.argv[1:]
sys.exit(main())
