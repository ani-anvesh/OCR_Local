#!/usr/bin/env python3
"""
Collect detailed system, GPU, CUDA, and ML-runtime info on Jetson devices.
Tested on Jetson Orin Nano / Xavier / Nano (JetPack 5.x/6.x)
"""

import os, json, subprocess, tarfile, time, platform, shutil, sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTBASE = PROJECT_ROOT / "tmp"
OUTBASE.mkdir(parents=True, exist_ok=True)
OUTDIR = OUTBASE / f"jetson_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTDIR.mkdir(parents=True, exist_ok=True)
print(f"[*] Saving diagnostics to: {OUTDIR}")

def run_cmd(cmd: str, file: str):
    """Run a shell command and save stdout/stderr to a file."""
    with open(OUTDIR / file, "w") as f:
        try:
            subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                           check=False, text=True)
        except Exception as e:
            f.write(f"[ERROR] {e}\n")

def write_text(name: str, content: str):
    (OUTDIR / name).write_text(content or "")

# --- System / OS ---
run_cmd("uname -a", "uname.txt")
run_cmd("lsb_release -a", "lsb_release.txt")
run_cmd("cat /etc/os-release", "os_release.txt")
run_cmd("cat /etc/nv_tegra_release", "nv_tegra_release.txt")
run_cmd("tr '\\0' '\\n' < /sys/firmware/devicetree/base/model", "device_tree_model.txt")

# --- CPU / Memory / Disk ---
run_cmd("cat /proc/cpuinfo", "cpuinfo.txt")
run_cmd("free -h", "memory.txt")
run_cmd("df -h", "disk_usage.txt")
run_cmd("lsblk -o NAME,SIZE,MODEL,MOUNTPOINT", "lsblk.txt")

# --- Packages ---
run_cmd("dpkg -l | grep -Ei 'nvidia|cuda|tensorrt|tegra'", "dpkg_nvidia.txt")

# --- CUDA / cuDNN / TensorRT ---
run_cmd("nvcc --version", "nvcc_version.txt")
run_cmd("cat /usr/local/cuda/version.txt", "cuda_version.txt")
run_cmd("ldconfig -p | grep -i cudnn", "ldconfig_cudnn.txt")
run_cmd("ldconfig -p | grep -i nvinfer", "ldconfig_tensorrt.txt")

# --- Jetson tools ---
run_cmd("sudo nvpmodel -q", "nvpmodel_q.txt")
run_cmd("sudo nvpmodel -m", "nvpmodel_m.txt")
run_cmd("(tegrastats --interval 1000 > /tmp/tegrastats_tmp.txt &) ; pid=$!; sleep 6; kill $pid; cat /tmp/tegrastats_tmp.txt", "tegrastats_sample.txt")

# --- Python environment ---
pyinfo = {
    "python_version": sys.version,
    "platform": platform.platform(),
}
try:
    import torch
    pyinfo["torch_version"] = torch.__version__
    pyinfo["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        pyinfo["device_name"] = torch.cuda.get_device_name(0)
        pyinfo["cuda_version"] = torch.version.cuda
except Exception as e:
    pyinfo["torch_error"] = str(e)

try:
    import paddle
    pyinfo["paddle_version"] = paddle.__version__
    pyinfo["paddle_device"] = paddle.get_device()
except Exception as e:
    pyinfo["paddle_error"] = str(e)

write_text("python_env.json", json.dumps(pyinfo, indent=2))
run_cmd("pip list", "pip_list.txt")

# --- GPU driver info ---
if Path("/proc/driver/nvgpu").exists():
    gpu_dir = OUTDIR / "proc_driver_nvgpu"
    gpu_dir.mkdir(exist_ok=True)
    for f in Path("/proc/driver/nvgpu").glob("*"):
        if f.is_file():
            shutil.copy(f, gpu_dir / f.name)

# --- Logs ---
run_cmd("dmesg | tail -n 200", "dmesg_tail.txt")
run_cmd("journalctl --no-pager -n 200 | egrep -i 'nvgpu|tegra|tensorrt|cudnn'", "journal_recent.txt")

# --- Package up results ---
tar_path = OUTDIR.with_suffix(".tar.gz")
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(OUTDIR, arcname=OUTDIR.name)

print(f"\n✅ Done! Logs saved to:\n  {OUTDIR}\n  {tar_path}")
