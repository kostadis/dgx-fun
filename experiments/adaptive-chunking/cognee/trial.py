"""Run both full-book arms sequentially and export aggregate metrics."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--python", type=Path, default=Path("/home/kostadis/cognee-local/.venv/bin/python"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-code-sha")
    p.add_argument("--adaptive-context", choices=("breadcrumb", "raw"), default="breadcrumb")
    p.add_argument("--llm-concurrency", type=int, choices=range(1, 9), default=4)
    a = p.parse_args()
    root = a.output.resolve()
    if not root.is_relative_to(HERE.parent / "raw"):
        p.error("Output must be under ignored raw/")
    if not a.resume and any((root / arm).exists() for arm in ("current", "adaptive")):
        p.error("Arm output already exists; use --resume or an unused output directory")
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "frozen"
    frozen.mkdir(exist_ok=True)
    source = HERE.parent / "raw/pabtso"
    inputs = {"corpus": source / "corpus.json", "cases": source / "cases.json",
              "chunks": source / "pilot-20261120/chunks-private.json",
              "selection": source / "pilot-20261120/intrinsic.json"}
    for key, path in inputs.items():
        destination = frozen / (key + ".json")
        if not destination.exists():
            shutil.copyfile(path, destination)
        inputs[key] = destination
    status = {"started_unix": time.time(), "stage": "starting", "passed": False,
              "adaptive_context": a.adaptive_context}
    status["llm_concurrency"] = a.llm_concurrency
    def save():
        (root / "control.json").write_text(json.dumps(status, indent=2) + "\n")
    for arm in ("current", "adaptive"):
        status["stage"] = arm
        save()
        args = [str(a.python), str(HERE / "run.py"), "--arm", arm, "--output", str(root / arm)]
        args.extend(["--llm-concurrency", str(a.llm_concurrency)])
        if arm == "adaptive":
            args.extend(["--adaptive-context", a.adaptive_context])
        for key, path in inputs.items():
            args.extend(["--" + key, str(path)])
        if a.resume:
            args.append("--resume")
        if a.resume_code_sha and (root / arm / "manifest.json").exists():
            args.extend(["--resume-code-sha", a.resume_code_sha])
        with (root / (arm + ".log")).open("a") as log:
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(stage="failed", failed_arm=arm, returncode=result.returncode,
                          finished_unix=time.time())
            save()
            raise SystemExit(result.returncode)
    status["stage"] = "analysis"
    save()
    subprocess.run([str(a.python), str(HERE / "analyze.py"), "--run", str(root),
                    "--cases", str(inputs["cases"]),
                    "--output", str(root / "metrics.json")], check=True)
    status.update(stage="complete-awaiting-answer-review", passed=True, finished_unix=time.time())
    save()


if __name__ == "__main__":
    main()
