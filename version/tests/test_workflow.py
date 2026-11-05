"""Test localhost routes with Flask's test client. Set TSOUDA_REAL_RVC=1 for CPU E2E."""
import io
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import numpy as np
import soundfile as sf

import app as ui
import engine as e


def vowel():
    t = np.arange(72000) / 32000
    phase = np.cumsum(220 + 45 * np.sin(t * 4)) * 2 * np.pi / 32000
    signal = sum(np.sin(k * phase) / k for k in range(1, 9))
    waveform = signal * 0.17 * np.minimum(1, np.minimum(t, 2.25 - t) * 25)
    stream = io.BytesIO()
    sf.write(stream, waveform, 32000, format="WAV")
    return stream.getvalue()


class Workflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        data = patch.object(e, "DATA", Path(self.temp.name))
        data.start()
        self.addCleanup(data.stop)
        self.addCleanup(ui.JOBS.clear)
        self.client = ui.app.test_client()
        self.clip = vowel()
        self.key = self.singer("測試歌手甲")

    def singer(self, name):
        response = self.client.post("/singers", data=dict(csrf=ui.CSRF, name=name))
        self.assertEqual(response.status_code, 303)
        return parse_qs(urlsplit(response.location).query)["singer"][0]

    def post(self, action, **data):
        return self.client.post(f"/singers/{self.key}/{action}", data=dict(csrf=ui.CSRF, **data),
                                follow_redirects=True)

    def wav(self, name="測試歌聲.WAV"):
        return io.BytesIO(self.clip), name

    def test_multiple_singers_and_wav_checks(self):
        second = self.singer("測試歌手乙")
        response = self.post("import", files=[self.wav(), (io.BytesIO(b"bad"), "bad.wav")])
        self.assertIn("匯入 1 檔", response.text)
        self.assertIn("bad.wav", response.text)
        self.assertIn("測試歌聲.WAV", response.text)
        self.assertIn("相同 WAV 已匯入", self.post("import", files=self.wav()).text)
        self.assertEqual(len(e.dataset_files(e.singer_path(second))), 0)
        wav = e.dataset_files(e.singer_path(self.key))[0]
        self.assertIn("已移除", self.post("remove", file=wav.stem).text)
        self.assertFalse(wav.exists())

    def test_reject_remote_host_or_missing_csrf(self):
        self.assertEqual(self.client.get("/", headers={"Host": "outside.example"}).status_code, 403)
        self.assertEqual(self.client.post("/singers", data={"name": "拒絕"}).status_code, 403)

    def test_oom_retries_at_lower_batch(self):
        path = e.singer_path(self.key)
        run = path / "run"
        weights = path / "weights"
        run.mkdir()

        def execute(*args, **options):
            self.assertEqual(args[0], "train.train")
            self.assertTrue(options["permit_oom"])
            batch = args[args.index("-bs") + 1]
            if batch == 2:
                return False
            self.assertEqual(batch, 1)
            for target in (run / "G_1.pth", run / "D_1.pth",
                           weights / "run_T1_model.pth"):
                target.touch()
            return True

        job = e.Job("OOM test", path / "test.log")
        with (patch.object(e, "require_assets"), patch.object(e, "prepared_run", return_value=run),
              patch.object(e, "WEIGHTS", weights), patch.object(job, "execute", side_effect=execute) as worker):
            e.train(job, self.key, 1, 1, 2, "gpu", "cpu", True)
        self.assertEqual(worker.call_count, 2)
        self.assertIn("Batch 降為 1", job.log.read_text(encoding="utf-8"))

    def test_pause_request_and_model_names(self):
        self.assertIn("目前沒有可暫停", self.post("pause").text)
        path = e.singer_path(self.key)
        job = e.Job("RVC v2 訓練", path / "test.log")
        job.pause_path = path / "test.pause"
        job.note("test")
        ui.JOBS[self.key] = job
        self.assertTrue(self.client.get(f"/status/{self.key}").json["can_pause"])
        self.post("pause")
        result = self.client.get(f"/status/{self.key}").json
        self.assertTrue(result["pause_requested"])
        self.assertTrue(job.pause_path.is_file())
        self.assertTrue(job.active)
        self.assertEqual(e.model_label("singer_batch_T20_model.pth"), "T20_model.pth")
        self.assertEqual(e.model_label("singer_batch_e20_s200.pth"), "T20_model.pth")

    def wait_job(self, status="完成"):
        for _ in range(2400):
            result = self.client.get(f"/status/{self.key}").json
            if not result["active"]:
                self.assertEqual(result["status"], status, result["lines"])
                return result
            time.sleep(0.25)
        self.fail("RVC job timed out")

    @unittest.skipUnless(os.environ.get("TSOUDA_REAL_RVC") == "1", "requires official weights and FFmpeg")
    def test_real_rvc_cpu_through_web_routes(self):
        e.require_assets()
        self.addCleanup(self.remove_rvc_outputs)
        self.assertIn("匯入 1 檔", self.post("import", files=self.wav()).text)
        self.post("prepare", features="cpu", fallback="on")
        self.wait_job()
        run = e.prepared_run(self.key)
        self.assertIsNotNone(run)
        self.post("train", epochs=3, interval=3, batch=1, device="cpu", features="cpu",
                  fallback="on")
        for _ in range(400):
            if self.client.get(f"/status/{self.key}").json.get("can_pause"):
                break
            time.sleep(0.05)
        else:
            self.fail("Pause did not become available")
        self.post("pause")
        result = self.wait_job("已暫停")
        self.assertEqual(result["progress"], [1, 3])
        self.assertIn("disc", result["losses"])
        self.assertEqual(e.latest_epoch(run), 1)
        model = f"{run.name}_T1_model.pth"
        with self.client.get(f"/models/{self.key}/{model}") as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("filename=T1_model.pth", response.headers["Content-Disposition"])
        self.post("convert", model=model, pitch=0, index=35, device="cpu",
                  vocal=self.wav(), fallback="on")
        self.wait_job()
        page = self.client.get(f"/?singer={self.key}").text
        self.assertIn("在同一時間點切換 A/B", page)
        folder = next((e.singer_path(self.key) / "conversions").iterdir())
        url = f"/audio/{self.key}/{folder.name}"
        with self.client.get(f"{url}/original", headers={"Range": "bytes=0-63"}) as response:
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.data[:4], b"RIFF")
        with self.client.get(f"{url}/converted?download=1") as response:
            info = sf.info(io.BytesIO(response.data))
            self.assertEqual((info.samplerate, info.channels, info.subtype), (32000, 1, "PCM_16"))
            self.assertLess(abs(info.duration - 2.25), 0.03)
            self.assertIn("attachment", response.headers["Content-Disposition"])

        self.post("train", epochs=3, interval=1, batch=1, device="cpu", features="cpu",
                  fallback="on")
        result = self.wait_job()
        self.assertEqual(result["progress"], [3, 3])
        self.assertEqual(e.latest_epoch(run), 3)
        self.assertIn("Resuming at epoch 2", ui.JOBS[self.key].log.read_text(encoding="utf-8"))
        self.assertFalse((run / "G_1.pth").exists())
        self.assertTrue((e.WEIGHTS / model).is_file())
        self.assertTrue((e.WEIGHTS / f"{run.name}_T3_model.pth").is_file())

    def remove_rvc_outputs(self):
        for run in e.run_dirs(self.key):
            shutil.rmtree(run)
        for model in e.model_files(self.key):
            model.unlink()


if __name__ == "__main__":
    unittest.main()
