"""xVGAE (NeurIPS 2026): a hierarchical variational graph autoencoder for exchangeable graphs."""
import os

__version__ = "1.0.0"
_limits = None


def set_threads(n=None):
    """Limit torch, BLAS and OpenMP to n CPU threads (default $XVGAE_THREADS, $OMP_NUM_THREADS, else min(4, CPUs))."""
    global _limits
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    n = int(n or os.environ.get("XVGAE_THREADS") or os.environ.get("OMP_NUM_THREADS") or min(4, cpus))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XVGAE_THREADS"):
        os.environ[v] = str(n)
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(n)
    _limits = threadpool_limits(n)
    return n


def pick_device(device=None):
    """torch.device: `device`, else cuda when torch sees a GPU, else cpu."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {device}: torch {torch.__version__} (CUDA {torch.version.cuda}) sees no GPU; install the "
                         f"wheel that matches the driver (e.g. --index-url https://download.pytorch.org/whl/cu128) or use cpu")
    return torch.device(device)


def atomic_save(obj, path):
    """torch.save to a temporary file, then rename it (a checkpoint is never left half written)."""
    import torch
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path):
    """json.dump to a temporary file, then rename it."""
    import json
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def file_sha1(path):
    """sha1 of a file (ties a z0 DDPM to the stage-1 checkpoint it was trained on)."""
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_id(path):
    """(size, mtime_ns) of a file: a cached score is reused only for the same file."""
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]
