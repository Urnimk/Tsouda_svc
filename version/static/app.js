const statusUrl = document.body.dataset.status;
let wasActive = document.body.dataset.active === "1";
async function poll() {
  try {
    const response = await fetch(statusUrl, { cache: "no-store" });
    if (!response.ok) return;
    const data = await response.json();
    document.querySelector("#gpu").textContent = data.gpu || "";
    if (!data.label) return;
    const pauseForm = document.querySelector("#pause-form");
    const pauseButton = document.querySelector("#pause-training");
    pauseForm.hidden = !data.can_pause;
    pauseButton.disabled = !!data.pause_requested;
    document.querySelector("#state").textContent = data.label + " · " + data.status;
    if (data.progress) {
      const [value, total] = data.progress;
      const progress = document.querySelector("#progress");
      progress.hidden = false;
      progress.max = total;
      progress.value = value;
      document.querySelector("#epoch-progress").textContent = `　Epoch ${value} / ${total}`;
    }
    const log = document.querySelector("#log");
    if (log.textContent !== data.lines) {
      log.textContent = data.lines;
      log.scrollTop = log.scrollHeight;
    }
    const losses = Object.entries(data.losses || {});
    if (losses.length) {
      document.querySelector("#loss").textContent =
        losses.map(([key, value]) => key.toUpperCase() + " " + value).join("　");
    }
    if (wasActive && !data.active) {
      window.location.reload();
      return;
    }
    wasActive = data.active;
  } catch {
    document.querySelector("#state").textContent = "暫時無法讀取工作進度";
  } finally {
    window.setTimeout(poll, 1800);
  }
}
if (statusUrl) poll();

for (const button of document.querySelectorAll("[data-remove]")) {
  button.addEventListener("click", (event) => {
    if (!window.confirm("從這個歌手的訓練資料集移除此片段？")) event.preventDefault();
  });
}

const model = document.querySelector("#model");
if (model) model.addEventListener("change", () => {
  document.querySelector("#index-rate").value = model.selectedOptions[0].dataset.index === "1" ? 35 : 0;
});

for (const audio of document.querySelectorAll("audio")) {
  audio.addEventListener("play", () => {
    for (const other of document.querySelectorAll("audio")) if (audio !== other) other.pause();
    audio.closest(".comparison").dataset.last = audio.src;
  });
}

for (const button of document.querySelectorAll(".switch")) {
  button.addEventListener("click", async () => {
    const card = button.closest(".comparison");
    const [a, b] = card.querySelectorAll("audio");
    const [from, to] = card.dataset.last === b.src ? [b, a] : [a, b];
    const position = from.currentTime;
    from.pause();
    if (Number.isFinite(to.duration)) to.currentTime = Math.min(position, Math.max(0, to.duration - 0.02));
    try { await to.play(); } catch { button.textContent = "請先用播放器載入音檔"; }
  });
}
