# Tsouda_svc V1.0

Tsouda_svc 是 Windows 本機歌聲轉換工作台。每個歌手建立獨立的資料集、訓練批次和模型。瀏覽器介面是繁體中文，透過 `127.0.0.1` 操作。第一次安裝會下載 Python 套件及 RVC 官方資源，之後訓練和轉換都在本機運算，無需付費 API。

## 從原本的 V1.0 升級

關閉程式後，將新版的 `app.py`、`engine.py`、`rvc/train/preprocess.py`、`rvc/train/train.py`、`templates/index.html`、`static/app.js`、`static/style.css` 複製至原專案的相同位置並取代，然後執行原本的 `start_windows.bat`。也可以把新版資料夾的內容覆蓋到原專案。已匯入資料及模型可繼續使用。

## 一、RTX 2060 相容性

| 項目 | V1.0 選擇 |
| --- | --- |
| 顯卡架構 | RTX 2060 採 Turing，CUDA Compute Capability 為 7.5（`sm_75`） |
| 系統與 Python | Windows 10／11 x64，Python 3.12 x64，可與其他 Python 版本並存 |
| CUDA 執行環境 | PyTorch 和 torchaudio 2.7.1，使用官方 CUDA 11.8（cu118）Windows wheel |
| 驅動程式 | 使用仍支援 RTX 2060 的 NVIDIA 驅動。CUDA 11.8 GA 對應的 Windows 驅動版本為 522.06 以上。安裝套件已含 CUDA runtime，通常只要 NVIDIA 驅動 |
| 6 GB 左右 | 預設 Batch 2、CPU 特徵提取、關閉 GPU 資料快取。若不穩定，改 Batch 1 |
| 12 GB 左右 | 偵測到 10 GiB 以上時預設 Batch 4，可按實際可用顯存調低 |
| 建議空間 | 至少 15 GB 可用空間作為初始安裝與單歌手測試的起點。多歌手、長資料集及多批次須保留更多空間 |

先在命令提示字元確認顯卡型號、總顯存、可用顯存與驅動：

```bat
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
```

RTX 2060 有不同顯存配置，請以 `memory.total` 為準。GPU 用量和可用顯存會受到 Windows 顯示、遊戲、錄影及其他 CUDA 程式影響。Batch 值是試跑起點，實際容量以本機一輪測試為準。

## 二、安裝與啟動

1. 把 ZIP **完整解壓縮**到可寫入的短路徑，例如 `D:\Tsouda_svc`。可以保留中文音檔名稱，程式會在匯入時轉為內部檔名。
2. 雙擊 `setup_windows.bat`。已有 Python 3.12 x64 時會直接使用，缺少時會嘗試以 winget 安裝。如無 winget，先由 [Python 官網](https://www.python.org/downloads/windows/) 安裝 3.12 x64，安裝時啟用 Python Launcher，再重跑腳本。
3. 安裝會建立 `.venv`，安裝 cu118 版 PyTorch、torchaudio 和 Python 套件，下載及驗證 HuBERT、RMVPE、32 kHz RVC v2 預訓練權重、FFmpeg 與 FFprobe。RVC 資源約 0.76 GB，另須下載 PyTorch 等套件。中斷後可重跑，已驗證的資源會直接沿用。
4. 安裝後會執行 CUDA FP16 矩陣乘法測試，接著開啟 `http://127.0.0.1:7862/`。日後直接雙擊 `start_windows.bat`。若瀏覽器未自動開啟，手動輸入網址。
5. 若 CUDA 測試失敗，介面會提供 CPU 裝置。執行 `diagnose_windows.bat` 查看實際顯卡、顯存、驅動及 PyTorch 輸出。要重試 GPU，更新驅動並重新啟動 Windows 後再診斷。

### 安裝排查

| 情況 | 處理方式 |
| --- | --- |
| `No suitable Python runtime found` | 安裝 Python 3.12 **64 位元**，用 `py -0p` 查看版本。已安裝 3.14 可保留，另裝 3.12 即可。 |
| `WinError 126`／DLL 載入失敗 | 安裝 [Microsoft Visual C++ x64 Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)，再重跑安裝。 |
| Hugging Face 或 PyTorch 下載失敗 | 確認可連線至 Hugging Face、PyPI 及 PyTorch 套件站。重跑安裝腳本可繼續。 |
| 埠 7862 已被使用 | 在專案目錄開啟命令提示字元，執行 `start_windows.bat --port 7863`，瀏覽 `http://127.0.0.1:7863/`。每個專案資料夾請只啟動一個服務實例。 |
| 要重建 Python 環境 | 關閉工作台後，將 `.venv` 改名為 `.venv_old`，再執行安裝。 |
| 音檔無法開啟 | 用音訊編輯器匯出為 16-bit PCM WAV，44.1 或 48 kHz，單聲道，再匯入。 |

## 三、資料準備與訓練

1. 新增一個歌手名稱，切換到該歌手，匯入已切好的 WAV。每位歌手使用自己有權使用的資料。
2. 優先使用清楚的單人乾聲，包含不同音高、母音和唱法。若素材混有伴奏、和聲或強烈殘響，先在外部清理。介面的品質欄檢查格式、時間長度、音量、近滿刻度取樣及聲道，音樂、串音、混響及人聲純度須試聽判斷。
3. 訓練檔接受 1～60 秒、16～192 kHz 的單聲道或立體聲 WAV，建議 2～15 秒。訓練切片匯入不設單檔或整批容量上限，也不限制每批檔案數。請確認系統暫存磁碟及專案所在磁碟有足夠空間。內容相同的 WAV 會依 SHA-256 去重。刪除片段後再訓練會使用新的資料快照。
4. 可先按 **只做預處理**。RVC 會進行高通、靜音分段、切片、重取樣與正規化，接著用 RMVPE 取 F0、用 HuBERT 取 v2 的 768 維特徵，並建立 FAISS 索引。直接開始訓練也會準備所需資料。
5. 先把 **總 Epoch** 和 **每幾輪保存** 都設為 **1**，選 GPU 0，Batch 設 1 或 2，跑完一次來確認本機顯存、Loss 和存檔。成功後可以把總 Epoch 調高，例如 100，保存間隔改為 10 或 20。同資料、同批次會從最近一組完整 G/D 檢查點的下一輪繼續。
6. Epoch 進度列、Loss 和最近紀錄會在頁面更新。Loss 紀錄可下載，另可用 TensorBoard 讀取 `rvc/logs`。生成器／判別器 Loss 是訓練訊號，挑模型時請以保留的演唱測試片段實際試聽。
7. 每到保存間隔和訓練結束時輸出可轉換的 Epoch 模型。大型 G/D 檢查點只保留每個批次最近兩組完整檔案，保留各次保存的推論模型。重新執行「只做預處理」會建立新批次，先前模型仍可選用。

### GPU 記憶體不足

- **訓練**勾選自動降級時，辨識到 CUDA OOM 會將 Batch 減半，最低為 1，並由最近一組完整檢查點重試。沒有檢查點時由預訓練權重重跑。
- **特徵提取**可手動選 CPU，這也是預設。若改選 GPU 且開啟自動降級，辨識到 OOM 時會切回 CPU。
- **聲音轉換**偵測到 CUDA OOM 可切回 CPU。長音檔可先分段試跑。
- Batch 1 仍 OOM 時，先關閉占用 GPU 的程式，查 `nvidia-smi` 的可用顯存，再以 Batch 1 續訓。介面也可選 CPU 訓練，時間依 CPU 與資料量而異。

### 暫停訓練後先試聽

進入模型訓練後，點「暫停訓練」。系統會跑完當前 Epoch，即使還沒到設定的保存間隔，也會保存檢查點與推論模型。等狀態顯示「已暫停」後，即可選擇模型進行轉換。要接著訓練，選同一位歌手、設定所需的總 Epoch，再按「開始／續訓」。若保存於第 20 輪，模型會顯示並下載為 `T20_model.pth`，續訓從第 21 輪開始。專案內部檔名附批次識別，以保留同輪數的不同模型與索引配對。

## 四、轉換與 A/B 試聽

1. 選擇歌手及想試聽的 Epoch 模型，上傳自己的演唱 WAV。轉換接受 0.2 秒到 15 分鐘及最多 120 MiB 的 WAV。
2. 先用 **移調 0** 和 **索引比例 35%**。RMVPE 提取原演唱音高，RVC 用此曲線引導合成並沿用演唱的時間排列。音高辨識及合成可能在氣音、假音、顫音或段落尾端產生偏差。
3. 完成後可播放原始演唱 A 和轉換 B，或用 **在同一時間點切換 A/B** 從接近同一位置換邊試聽。輸出可下載為 32 kHz、16-bit PCM 單聲道 WAV。
4. 可在「查看／下載訓練模型」下載 `.pth`。每個批次的 `added_*.index` 在對應 `rvc/logs/<批次>` 資料夾。備份模型時，保留兩者的配對關係。

| 路徑 | 內容 |
| --- | --- |
| `workspace/<歌手 ID>/dataset/` | 原始 WAV 和品質檢查結果 |
| `workspace/<歌手 ID>/jobs/` | 工作文字紀錄 |
| `workspace/<歌手 ID>/conversions/` | 原唱、輸出及轉換參數 |
| `rvc/logs/<批次>/` | 預處理、F0、特徵、索引、TensorBoard、續訓檢查點 |
| `rvc/assets/weights/` | 各 Epoch 和最新推論模型 |

## 五、測試與來源

- `.venv\Scripts\python.exe -m unittest discover -s tests -v` 驗證多歌手隔離、有效與無效 WAV、去重、移除、本機路由及模擬 CUDA OOM 時的 Batch 降級。
- 先在 CMD 執行 `set TSOUDA_REAL_RVC=1`，再跑同一指令，可用合成母音做 CPU 端到端驗證，涵蓋預處理、RMVPE、HuBERT、索引、3 個 Epoch、續訓、模型及結果 WAV 下載。測試使用獨立歌手 ID，結束後清理該測試輸出。測試期間請保留至少約 5 GB 可用空間。
- 交付測試使用 Linux CPU 與合成母音。Windows x64 依賴有做 wheel 解析，A/B 的音源及位元組範圍請求已測試，前端切換邏輯以模擬播放器測試。Windows 批次腳本、瀏覽器實際音訊播放、實機 CUDA 訓練、真實演唱音色與顯存占用須在使用者電腦驗證。
- RVC 核心以 [官方原始碼 81eed5e](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/81eed5e8f68b6bed1789f682fe78cdd324495afc) 為基礎，附上游授權。專案調整了單裝置訓練、檢查點的原子寫入與配對續訓、檢查點保留和進度訊號。
- [RVC 官方 CLI 文件](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/cli.md) · [RVC 資源版本](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/e6d0c1a17da07c33557852f9dfa2bd44cc75737d) · [PyTorch 歷史版本](https://docs.pytorch.org/get-started/previous-versions/) · [NVIDIA CUDA 11.8 發行說明](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/) · [NVIDIA 顯卡規格](https://www.nvidia.com/zh-tw/geforce/graphics-cards/compare/?section=compare-specs)
