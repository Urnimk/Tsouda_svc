"""Local WAV library and orchestration of the pinned RVC CLI."""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid

import numpy as np
import soundfile as sf

from bootstrap import ROOT, RVC, ASSETS

DATA = ROOT / "workspace"
WEIGHTS = RVC / "assets" / "weights"
LIMIT = 120 * 1024**2
OOM = re.compile(r"cuda.{0,60}out of memory|cudnn_status_alloc_failed|cublas_status_alloc_failed|torch\.OutOfMemoryError", re.I)
LOSS = re.compile(r"loss_(disc|gen|fm|mel|kl)=([^,\s]+)")


def token():
    return uuid.uuid4().hex[:16]


def save_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def singer_path(key):
    if not re.fullmatch(r"[a-f0-9]{16}", key) or not (DATA / key / "singer.json").is_file():
        raise ValueError("找不到歌手專案")
    return DATA / key


def create_singer(name):
    name = name.strip()
    if not 1 <= len(name) <= 64:
        raise ValueError("歌手名稱須為 1～64 字元")
    key = token()
    path = DATA / key
    (path / "dataset").mkdir(parents=True)
    (path / "conversions").mkdir()
    save_json(path / "singer.json", {"name": name})
    return key


def inspect_wav(path, dataset=True):
    with sf.SoundFile(path) as audio:
        rate, channels = audio.samplerate, audio.channels
        seconds = len(audio) / rate
        if audio.format not in {"WAV", "WAVEX", "RF64"}:
            raise ValueError("檔案內容須為 WAV")
        if channels not in (1, 2) or not 16000 <= rate <= 192000:
            raise ValueError("須為單聲道或立體聲，取樣率 16～192 kHz")
        if not ((1.0 if dataset else 0.2) <= seconds <= (60 if dataset else 900)):
            raise ValueError("訓練片段須為 1～60 秒" if dataset else "轉換音檔須為 0.2 秒～15 分鐘")
        peak = energy = 0.0
        clipped = 0
        for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError("音檔含有無效的取樣值")
            peak = max(peak, float(np.max(np.abs(block))))
            clipped += int(np.count_nonzero(np.abs(block) >= 0.999))
            mono = block.mean(axis=1, dtype=np.float64)
            energy += float(mono @ mono)
        rms = float(np.sqrt(energy / len(audio)))
        if rms < 0.0001:
            raise ValueError("音檔接近靜音，或左右聲道合成後抵銷")
        warnings = []
        if clipped:
            warnings.append(f"近滿刻度取樣 {clipped:,} 點，可能削波")
        if rms < 0.01:
            warnings.append("音量偏低")
        if channels == 2:
            warnings.append("RVC 會混為單聲道")
        if dataset and not 2 <= seconds <= 15:
            warnings.append("建議切成 2～15 秒片段")
        return dict(rate=rate, channels=channels, seconds=round(seconds, 2),
                    peak=round(peak, 4), rms_db=round(20 * np.log10(rms), 1),
                    subtype=audio.subtype, warnings=warnings)


def store_wav(upload, folder, dataset=True):
    if Path(upload.filename).suffix.lower() != ".wav":
        raise ValueError("請選擇 .wav 檔")
    temp = folder / (token() + ".part")
    try:
        upload.save(temp)
        if not dataset and temp.stat().st_size > LIMIT:
            raise ValueError("單檔上限為 120 MiB")
        info = inspect_wav(temp, dataset)
        if dataset:
            with temp.open("rb") as stream:
                key = hashlib.file_digest(stream, "sha256").hexdigest()
            target = folder / f"{key}.wav"
            if target.exists():
                raise ValueError("相同 WAV 已匯入")
            info["name"] = upload.filename
            save_json(folder / f"{key}.json", info)
        else:
            target = folder / "original.wav"
        temp.replace(target)
        return info
    finally:
        temp.unlink(missing_ok=True)


def dataset_files(path):
    return sorted((path / "dataset").glob("*.wav"))


def run_dirs(key):
    return sorted((RVC / "logs").glob(f"{key}_*"), key=lambda path: path.stat().st_mtime, reverse=True)


def model_files(key):
    return sorted(WEIGHTS.glob(f"{key}_*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)


def model_label(name):
    match = re.search(r"_(?:T(\d+)_model|e(\d+)_s\d+)\.pth$", name)
    if not match:
        return name
    return f"T{match[1] or match[2]}_model.pth"


def find_index(model):
    run = model.stem.split("_")[:2]
    if len(run) == 2:
        return next((RVC / "logs" / "_".join(run)).glob("added_*.index"), None)
    return None


@dataclass
class Job:
    label: str
    log: Path
    active: bool = True
    status: str = "進行中"
    result: str = ""
    progress: tuple[int, int] | None = None
    pause_path: Path | None = None
    paused_epoch: int | None = None
    losses: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def note(self, message):
        with self.lock, self.log.open("a", encoding="utf-8") as stream:
            stream.write(str(message) + "\n")

    def execute(self, module, *args, cpu=False, permit_oom=False):
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", RVC_CUDA_GRAPH="0",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   PATH=str(RVC) + os.pathsep + env.get("PATH", ""))
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            env["RVC_AUDIO_FORCE_CPU"] = "1"
        else:
            env["CUDA_VISIBLE_DEVICES"] = "0"
        if module == "train.train" and self.pause_path is not None:
            env["TSOUDA_PAUSE_REQUEST"] = str(self.pause_path.resolve())
        command = [sys.executable, "-u", "-m", module, *map(str, args)]
        self.note(subprocess.list2cmdline(command))
        oom = False
        with subprocess.Popen(command, cwd=RVC, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace") as process:
            for line in process.stdout:
                self.note(line.rstrip())
                if "loss_disc=" in line:
                    self.losses = dict(LOSS.findall(line))
                oom |= bool(OOM.search(line))
                progress = re.search(r"TSOUDA_EPOCH (\d+)/(\d+)", line)
                if progress:
                    self.progress = tuple(map(int, progress.groups()))
                paused = re.search(r"TSOUDA_PAUSED (\d+)", line)
                if paused:
                    self.paused_epoch = int(paused[1])
        if oom and permit_oom:
            return False
        if process.returncode or oom:
            raise RuntimeError(f"{module} 執行失敗（代碼 {process.returncode}），請看完整紀錄")
        return True


def require_assets():
    missing = [entry["dest"] for entry in ASSETS.values()
               if (os.name == "nt" or not entry["dest"].endswith(".exe"))
               and not (RVC / entry["dest"]).is_file()]
    if missing:
        raise ValueError("請先執行 setup_windows.bat 安裝資源：" + ", ".join(missing))


def prepared_run(key):
    path = singer_path(key)
    names = [wav.name for wav in dataset_files(path)]
    for run in run_dirs(key):
        marker = run / "prepared.json"
        if marker.is_file() and json.loads(marker.read_text()) == names:
            return run
    return None


def prepare(job, key, device, auto_oom):
    require_assets()
    path = singer_path(key)
    wavs = dataset_files(path)
    if not wavs:
        raise ValueError("請先匯入 WAV")
    run = RVC / "logs" / f"{key}_{token()}"
    run.mkdir(parents=True)
    exp = run.relative_to(RVC).as_posix()
    cpu = device == "cpu"
    job.note("階段 1／4　人聲預處理")
    # CPU decode isolates input resampling from GPU memory fluctuations.
    job.execute("train.preprocess", path / "dataset", 32000, 1, exp, "True", 3.0, cpu=True)
    if not any((run / "1_16k_wavs").glob("*.wav")):
        raise RuntimeError("預處理未產生任何可用片段")
    job.note("階段 2／4　RMVPE 音高提取")
    if cpu or not job.execute("train.dataset.extract_f0", "cuda", 1, 0, 0, exp, "true",
                              permit_oom=auto_oom):
        if not cpu:
            job.note("GPU 記憶體不足，RMVPE 改用 CPU")
        job.execute("train.dataset.extract_f0", "cpu", exp, 1, "rmvpe", cpu=True)
    job.note("階段 3／4　HuBERT v2 特徵提取")
    if cpu or not job.execute("train.dataset.extract_hubert_feature", "cuda:0", 1, 0, 0, exp,
                              "v2", "true", permit_oom=auto_oom):
        if not cpu:
            job.note("GPU 記憶體不足，HuBERT 改用 CPU")
        job.execute("train.dataset.extract_hubert_feature", "cpu", 1, 0, exp, "v2", "false", cpu=True)
    rows = []
    for wav in sorted((run / "0_gt_wavs").glob("*.wav")):
        paths = [wav, run / "3_feature768" / f"{wav.stem}.npy",
                 run / "2a_f0" / f"{wav.name}.npy", run / "2b-f0nsf" / f"{wav.name}.npy"]
        # Match the official trainer's sample-length buckets.
        if not all(file.is_file() for file in paths) or not 100 < wav.stat().st_size // 960 <= 900:
            job.note(f"略過：{wav.name} 的特徵不完整或片段長度不符")
            continue
        rows.append("|".join([*(file.relative_to(RVC).as_posix() for file in paths), "0"]))
    if not rows:
        raise RuntimeError("沒有符合 RVC 訓練條件的片段，請使用清晰且約 2～15 秒的單人歌聲")
    (run / "filelist.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    config = json.loads((RVC / "configs" / "v2" / "32k.json").read_text())
    config["train"]["log_interval"] = 10
    save_json(run / "config.json", config)
    job.note(f"階段 4／4　FAISS 索引，訓練片段 {len(rows)} 個")
    job.execute("train.train_index", run.name, "v2", exp, 2, "single", cpu=True)
    if not any(run.glob("added_*.index")):
        raise RuntimeError("FAISS 未產生可用索引")
    save_json(run / "prepared.json", [wav.name for wav in wavs])
    return run


def latest_epoch(run):
    return max((int(p.stem[2:]) for p in run.glob("G_*.pth")
                if p.stem[2:].isdigit() and (run / f"D_{p.stem[2:]}.pth").is_file()), default=0)


def train(job, key, epochs, interval, batch, device, features, auto_oom):
    require_assets()
    run = prepared_run(key) or prepare(job, key, features, auto_oom)
    completed = latest_epoch(run)
    job.progress = (min(completed, epochs), epochs)
    if completed >= epochs:
        raise ValueError(f"已完成 {completed} 輪，請指定更高的總 Epoch 或重新預處理")
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    cpu = device == "cpu"
    job.pause_path = job.log.with_suffix(".pause")
    while True:
        job.note(f"訓練開始：{device.upper()}，Batch {batch}，目標 Epoch {epochs}")
        success = job.execute("train.train", "-e", run.name, "-sr", "32k", "-f0", 1,
                              "-bs", batch, "-g", "" if cpu else "0", "-te", epochs,
                              "-se", interval, "-pg", "assets/pretrained_v2/f0G32k.pth",
                              "-pd", "assets/pretrained_v2/f0D32k.pth", "-l", 0,
                              "-c", 0, "-sw", 1, "-v", "v2", cpu=cpu,
                              permit_oom=not cpu and auto_oom)
        if success:
            break
        if batch == 1:
            raise RuntimeError("Batch 1 仍顯示 CUDA 記憶體不足。關閉占用顯存的程式再續訓，或選 CPU。")
        batch = max(1, batch // 2)
        job.note(f"CUDA 記憶體不足，Batch 降為 {batch}，從最近一組完整檢查點重試")
    saved_epoch = job.paused_epoch or epochs
    final = WEIGHTS / f"{run.name}_T{saved_epoch}_model.pth"
    if latest_epoch(run) != saved_epoch or not final.is_file():
        raise RuntimeError("找不到本次訓練的檢查點和推論模型，請檢查訓練紀錄與磁碟容量")
    if job.paused_epoch is not None:
        job.result = f"已保存 T{saved_epoch}_model，可轉換音色或從第 {saved_epoch + 1} 輪續訓"


def convert(job, model, folder, pitch, rate, device, auto_oom):
    require_assets()
    index = find_index(model)
    if rate and index is None:
        raise ValueError("這個模型沒有配對索引，請將索引比例設為 0")
    temp = folder / "pending.wav"
    args = ["--model", model, "--input", folder / "original.wav", "--output", temp,
            "--pitch", pitch, "--f0-method", "rmvpe", "--index-rate", rate, "--overwrite"]
    if rate:
        args.extend(["--index", index])
    cpu = device == "cpu"
    if not job.execute("infer.cli", *args, cpu=cpu, permit_oom=auto_oom and not cpu):
        job.note("GPU 記憶體不足，改用 CPU 轉換")
        job.execute("infer.cli", *args, cpu=True)
    if not temp.is_file() or sf.info(temp).frames <= 0:
        raise RuntimeError("RVC 未輸出有效 WAV")
    temp.replace(folder / "converted.wav")
