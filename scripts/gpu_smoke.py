"""Small real tensor calculation; CUDA is mandatory unless explicitly allowed."""
import argparse
import json
import platform
from pathlib import Path


def select_device(cuda_available, allow_cpu=False):
    if cuda_available:
        return "cuda"
    if allow_cpu:
        return "cpu"
    raise RuntimeError(
        "CUDA is unavailable. Select a GPU accelerator in Kaggle and check "
        "account verification/quota. CPU fallback requires --allow-cpu."
    )


def run_smoke(allow_cpu=False):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing; use the Kaggle Python image or install a suitable PyTorch build.") from exc
    device = select_device(torch.cuda.is_available(), allow_cpu)
    # A small deterministic calculation, not a benchmark or a shogi model.
    x = torch.arange(256, dtype=torch.float32, device=device).reshape(16, 16)
    result = x @ x.T
    torch.testing.assert_close(result.cpu(), x.cpu() @ x.cpu().T)
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "status": "passed", "device": device,
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if device == "cuda" else 0,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "test": "16x16 float32 matrix multiplication against CPU reference",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_smoke(args.allow_cpu)
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
