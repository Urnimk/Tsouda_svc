"""Tsouda_svc localhost control panel."""
import argparse
import collections
import hmac
import json
import re
import sys
import secrets
import shutil
import subprocess
import threading
import webbrowser

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge

import engine as e

app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=None, MAX_FORM_PARTS=None)
CSRF = secrets.token_hex(24)
PORT = 7862
LOCK = threading.Lock()
JOBS = {}
GPU = {"ready": False, "name": "尚未偵測", "memory": 0}
GPU_USAGE = ""
GPU_ENV = threading.Event()


def gpu_monitor():
    global GPU_USAGE
    while not GPU_ENV.wait(2):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--id=0", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            util, used, total = [int(item.strip()) for item in result.stdout.strip().split(",")]
            GPU_USAGE = f"GPU {util}%　VRAM {used:,}／{total:,} MiB"
        except (OSError, ValueError, subprocess.TimeoutExpired):
            GPU_USAGE = "即時 GPU 遙測無法讀取"


def init_gpu():
    try:
        result = subprocess.run(
            [sys.executable, str(e.ROOT / "bootstrap.py"), "diagnose", "--json"],
            capture_output=True, text=True, encoding="utf-8", timeout=90)
        if result.returncode == 0:
            GPU.update(json.loads(result.stdout))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    if GPU["ready"]:
        threading.Thread(target=gpu_monitor, daemon=True).start()


@app.before_request
def local_only():
    if request.host not in {f"127.0.0.1:{PORT}", f"localhost:{PORT}", "localhost", "127.0.0.1"}:
        abort(403)
    is_slice_upload = (
        request.endpoint == "action" and (request.view_args or {}).get("action") == "import"
    )
    if not is_slice_upload:
        request.max_content_length = 512 * 1024**2
        request.max_form_parts = 512
    if request.method == "POST" and not hmac.compare_digest(request.form.get("csrf", ""), CSRF):
        abort(403)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_singers():
    return [(p.parent.name, read_json(p)["name"]) for p in sorted(e.DATA.glob("*/singer.json"))]


def view(key="", message=""):
    entries = load_singers()
    if not key and entries:
        key = entries[0][0]
    files, models, conversions, run = [], [], [], None
    if key:
        path = e.singer_path(key)
        for wav in e.dataset_files(path):
            meta = wav.with_suffix(".json")
            if meta.is_file():
                files.append((wav.stem, read_json(meta)))
        models = [(model.name, bool(e.find_index(model)), e.model_label(model.name))
                  for model in e.model_files(key)]
        labels = collections.Counter(label for _, _, label in models)
        models = [(name, index, f"{label}（批次 {name.split('_')[1][:8]}）"
                   if labels[label] > 1 else label) for name, index, label in models]
        conversions = [(p.parent.name, read_json(p)) for p in
                       sorted((path / "conversions").glob("*/result.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True)]
        run = e.prepared_run(key)
    logs = sorted((e.DATA / key / "jobs").glob("*.log"),
                  key=lambda p: p.stat().st_mtime, reverse=True)[:10] if key else []
    job = JOBS.get(key)
    return render_template("index.html", key=key, singers=entries, files=files,
                           duration=sum(meta["seconds"] for _, meta in files),
                           models=models, conversions=conversions, run=run,
                           epoch=e.latest_epoch(run) if run else 0, job=job,
                           gpu=GPU, csrf=CSRF, message=message, logs=logs)


@app.get("/")
def home():
    try:
        return view(request.args.get("singer", ""), request.args.get("message", ""))
    except ValueError:
        abort(404)


def submit(key, label, work, *args):
    if any(job.active for job in JOBS.values()):
        raise ValueError("已有工作執行中，請等待完成")
    path = e.singer_path(key) / "jobs"
    path.mkdir(exist_ok=True)
    job = e.Job(label, path / f"{e.token()}.log")
    job.note(label)
    JOBS[key] = job

    def run():
        final_status = "失敗"
        try:
            work(job, *args)
        except Exception as error:
            job.note(f"失敗：{error}")
        else:
            final_status = "已暫停" if job.paused_epoch is not None else "完成"
            job.note(job.result or "完成")
        finally:
            with LOCK:
                job.status = final_status
                job.active = False
                if job.pause_path is not None:
                    job.pause_path.unlink(missing_ok=True)

    threading.Thread(target=run, daemon=True).start()


def int_field(name, low, high):
    value = int(request.form[name])
    if not low <= value <= high:
        raise ValueError(f"{name} 超出範圍")
    return value


def selected_device(field):
    value = request.form.get(field)
    if value == "cpu":
        return value
    if value == "gpu" and GPU["ready"]:
        return value
    raise ValueError("CUDA 尚未通過偵測。請先執行 diagnose_windows.bat，或選擇 CPU。")


def complete_conversion(job, model, folder, pitch, rate, device, fallback, original_name):
    try:
        e.convert(job, model, folder, pitch, rate, device, fallback)
        e.save_json(folder / "result.json", dict(name=original_name, model=e.model_label(model.name),
                                                pitch=pitch, rate=rate))
        job.result = "轉換完成，請到 A/B 比較區試聽"
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def flash(key, message=""):
    return redirect(url_for("home", singer=key, message=message), code=303)


@app.post("/singers")
def add_singer():
    try:
        with LOCK:
            return flash(e.create_singer(request.form.get("name", "")), "歌手專案已建立")
    except ValueError as error:
        return flash("", str(error))


@app.post("/singers/<key>/<action>")
def action(key, action):
    try:
        path = e.singer_path(key)
    except ValueError:
        abort(404)
    try:
        with LOCK:
            if action == "pause":
                job = JOBS.get(key)
                if job is None or not job.active or job.pause_path is None:
                    raise ValueError("目前沒有可暫停的模型訓練")
                job.pause_path.touch()
                job.status = "等待當前輪次完成並存檔"
                return flash(key)
            if any(job.active for job in JOBS.values()):
                raise ValueError("已有工作執行中，請等待完成")
            if action == "import":
                errors = []
                count = 0
                for upload in request.files.getlist("files"):
                    try:
                        e.store_wav(upload, path / "dataset")
                        count += 1
                    except (ValueError, OSError, RuntimeError) as error:
                        errors.append(f"{upload.filename}：{error}")
                return flash(key, f"匯入 {count} 檔" + ("。拒收 " + "；".join(errors[:8]) if errors else ""))
            if action == "remove":
                stem = request.form.get("file", "")
                if not re.fullmatch("[0-9a-f]{64}", stem):
                    raise ValueError("無效的資料檔案")
                (path / "dataset" / f"{stem}.wav").unlink()
                (path / "dataset" / f"{stem}.json").unlink(missing_ok=True)
                return flash(key, "已移除資料集中的片段")
            fallback = "fallback" in request.form
            if action == "prepare":
                submit(key, "人聲預處理", e.prepare, key, selected_device("features"), fallback)
            elif action == "train":
                submit(key, "RVC v2 訓練", e.train, key, int_field("epochs", 1, 1200),
                       int_field("interval", 1, 1200), int_field("batch", 1, 16),
                       selected_device("device"), selected_device("features"), fallback)
            elif action == "convert":
                models = {model.name: model for model in e.model_files(key)}
                model = models.get(request.form.get("model"))
                if model is None:
                    raise ValueError("請選擇已訓練的模型")
                rate = int_field("index", 0, 100) / 100
                pitch = int_field("pitch", -24, 24)
                device = selected_device("device")
                upload = request.files.get("vocal")
                if upload is None:
                    raise ValueError("請選擇演唱 WAV")
                folder = path / "conversions" / e.token()
                folder.mkdir()
                try:
                    e.store_wav(upload, folder, dataset=False)
                    submit(key, "RVC v2 歌聲轉換", complete_conversion,
                           model, folder, pitch, rate, device, fallback, upload.filename)
                except Exception:
                    shutil.rmtree(folder, ignore_errors=True)
                    raise
            else:
                abort(404)
    except (ValueError, RuntimeError, OSError) as error:
        return flash(key, str(error))
    return flash(key)


@app.get("/status/<key>")
def status(key):
    job = JOBS.get(key)
    if not job:
        return jsonify(active=False, gpu=GPU_USAGE)
    with job.lock, job.log.open("rb") as stream:
        stream.seek(max(0, job.log.stat().st_size - 128 * 1024))
        lines = collections.deque(stream.read().decode("utf-8", "replace").splitlines(), maxlen=60)
    return jsonify(label=job.label, active=job.active, status=job.status,
                   lines="\n".join(lines), progress=job.progress,
                   losses=job.losses, gpu=GPU_USAGE,
                   can_pause=job.active and job.pause_path is not None,
                   pause_requested=job.active and job.pause_path is not None and job.pause_path.exists())


@app.get("/logs/<key>/<log>")
def logfile(key, log):
    try:
        path = e.singer_path(key) / "jobs" / f"{log}.log"
        if re.fullmatch(r"[a-f0-9]{16}", log) and path.is_file():
            return send_file(path, as_attachment=True)
    except ValueError:
        pass
    abort(404)


@app.get("/models/<key>/<name>")
def model_download(key, name):
    try:
        e.singer_path(key)
    except ValueError:
        abort(404)
    matches = {path.name: path for path in e.model_files(key)}
    if name in matches:
        return send_file(matches[name], as_attachment=True, download_name=e.model_label(name))
    abort(404)


@app.get("/audio/<key>/<conversion>/<variant>")
def audio(key, conversion, variant):
    try:
        folder = e.singer_path(key) / "conversions" / conversion
    except ValueError:
        abort(404)
    if (not re.fullmatch(r"[a-f0-9]{16}", conversion)
            or variant not in ("original", "converted") or not (folder / "result.json").is_file()):
        abort(404)
    return send_file(folder / f"{variant}.wav", mimetype="audio/wav",
                     as_attachment=request.args.get("download") == "1", conditional=True)


@app.errorhandler(RequestEntityTooLarge)
def oversized(_):
    return "表單超過此操作允許的上限。", 413


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    PORT = args.port
    init_gpu()
    if not args.no_browser:
        threading.Timer(1, webbrowser.open, args=[f"http://127.0.0.1:{PORT}/"]).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
