"""Download and verify the official RVC assets and diagnose the runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parent
RVC = ROOT / "rvc"
REV = "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"
BASE = f"https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/{REV}/"
ASSETS = json.loads((ROOT / "assets_manifest.json").read_text())


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    if entry["sha256"]:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == entry["blob"]


def download(item):
    name, entry = item
    path = RVC / entry["dest"]
    if verify(path, entry):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    print(f"下載 {name} ({entry['size'] / 1e6:.1f} MB)", flush=True)
    try:
        with urllib.request.urlopen(BASE + name + "?download=true", timeout=90) as response:
            with temp.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        if not verify(temp, entry):
            raise RuntimeError(f"檔案大小或雜湊不符：{name}")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def diagnose(machine=False):
    import torch
    if not machine:
        print(f"PyTorch {torch.__version__} / CUDA runtime {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA 無法使用。請確認 NVIDIA 驅動程式及 cu118 版 PyTorch。")
        return False
    device = torch.cuda.get_device_properties(0)
    memory = round(device.total_memory / 2**30, 2)
    if not machine:
        print(f"{device.name} | VRAM {memory:.2f} GiB | SM {device.major}.{device.minor}")
    x = torch.ones((64, 64), device="cuda", dtype=torch.float16)
    assert (x @ x)[0, 0].item() == 64
    if machine:
        print(json.dumps(dict(ready=True, name=device.name, memory=memory)))
    else:
        print("CUDA FP16 矩陣運算通過。")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["assets", "diagnose"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.action == "diagnose":
        raise SystemExit(0 if diagnose(args.json) else 1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, [item for item in ASSETS.items()
                                 if os.name == "nt" or not item[0].endswith(".exe")]))
    print("RVC 資源驗證完成。")
