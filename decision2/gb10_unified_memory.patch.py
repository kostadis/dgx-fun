"""Patch vllm_sr_runtime/accel/gpu.py for integrated (unified-memory) GPUs such as the GB10.

torch.cuda.mem_get_info() on the GB10 reports free memory that excludes reclaimable
page cache, so a box with 34 GB available reports ~9-18 GiB "free" -- and reading the
weights to verify their checksums refills that cache right before placement checks it.
On an integrated device, use the kernel's MemAvailable instead. Discrete GPUs unchanged.
"""
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
old = "                free, total = torch.cuda.mem_get_info(index)\n"
new = (old +
"                if getattr(properties, 'is_integrated', False):\n"
"                    # GB10 / unified memory: CUDA's 'free' ignores reclaimable page cache.\n"
"                    for line in open('/proc/meminfo'):\n"
"                        if line.startswith('MemAvailable:'):\n"
"                            free = int(line.split()[1]) * 1024\n"
"                            break\n")
assert s.count(old) == 1, "gpu.py changed upstream; re-check this patch"
p.write_text(s.replace(old, new))
print("patched", p)
