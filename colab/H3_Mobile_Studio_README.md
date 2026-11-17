# H3 Mobile Studio｜Google Colab Pro 手機工作站

## 一鍵啟動
[直接在 Google Colab 開啟](https://colab.research.google.com/github/eric931130/aiagent/blob/main/colab/H3_Mobile_Studio_ColabPro_Launcher_20261117.ipynb)

1. 在 Colab 右上角切換 L4 或 A100 GPU。
2. 執行前兩格：安裝 Gradio、掛載 Google Drive、取得程式。
3. 執行第三格，輸入至少 10 字元的密碼。
4. Colab 輸出 **Running on public URL: https://xxxx.gradio.live**。
5. 用手機 Chrome 開啟該 URL，以帳號 `studio` 和自己設定的密碼登入。
6. 在 H3 圖生影片分頁上傳圖片並提交工作；在任務分頁查詢狀態、預覽及下載。
7. 需要後製時，進入畫質＋中文配音分頁，選擇已有任務或上傳 MP4。
8. 成品會備份到 `Google Drive/我的雲端硬碟/H3_Mobile_Studio`。

## 行動裝置
手機 Chrome 選擇「加入主畫面」或「安裝應用程式」，可用 PWA 方式開啟。不需要 APK。由於 Gradio share 網址是**暫時連結**，當 Colab 重啟或分享網址到期，需要重新取得新的網址；舊的主畫面捷徑會失效。

## 真正的後端位置
GPU 推論發生在正在執行的 Google Colab Pro VM，不是手機，也不是 GitHub。提交任務會排入 Colab 內部背景佇列；只要 Colab VM 仍存活，切換手機畫面不會主動停止已提交工作。執行階段到期、運算單元不足、網路中斷或後端錯誤仍會讓工作中斷。

## 工作站內容
- MiniMax H3 I2V：首幀圖片、中文/英文提示詞、5/10/15 秒、原生音效。
- Real-ESRGAN Anime AI upscaler：逐幀增強，720p/1080p。
- Qwen3-TTS 0.6B CustomVoice：中文台詞、聲線與情緒描述。
- FFmpeg：台詞字幕、混音及 MP4。
- FFmpeg minterpolate：可選的 **48fps 光流插幀（非 RIFE）**。
- 登入密碼保護 + 單一 GPU 工作佇列 + Google Drive 備份。

## 注意
- 首次下載的 H3 模型超過 35GB；Colab GPU 與 CPU RAM、磁碟容量需足夠，L4 不保證所有高畫質設定成功。
- 將 H3 影片做 720p/1080p 放大不等於重建丟失的真實細節。
- 字幕是根據自行輸入的中文台詞生成的整句字幕，非逐字強制對齊。
- 本專案已建立程式與 Notebook 並檢查檔案存在，但尚未在你的 Colab GPU 端到端實測。
- Google Colab Pro 不是 24/7 伺服器；需要永久網址、固定 GPU，請改用固定雲端 GPU 服務或自有主機。
- 生成內容的商用權利請參照 MiniMax H3 官方授權條款。

來源：
- https://research.google.com/colaboratory/faq.html
- https://gradio.app/guides/sharing-your-app
- https://docs.comfy.org/tutorials/video/minimax/minimax-h3
