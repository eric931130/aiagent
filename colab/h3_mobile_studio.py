# -*- coding: utf-8 -*-
"""H3 Mobile Studio: password-protected Gradio PWA backed by a Colab GPU.
For temporary interactive use; not a 24/7 server.
"""
import os, sys, json, time, uuid, gc, shutil, threading, subprocess, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fractions import Fraction

import requests
import gradio as gr
from PIL import Image, ImageOps

BASE = Path("/content/ComfyUI")
ROOT = Path("/content/H3_Mobile_Studio")
WORK = ROOT / "jobs"
PUBLIC = ROOT / "results"
LOG = ROOT / "comfyui.log"
DATA = ROOT / "jobs.json"
for d in (ROOT, WORK, PUBLIC):
    d.mkdir(parents=True, exist_ok=True)

DRIVE = Path("/content/drive/MyDrive/H3_Mobile_Studio")
DRIVE_READY = Path("/content/drive/MyDrive").exists()
if DRIVE_READY:
    DRIVE.mkdir(parents=True, exist_ok=True)

API = "http://127.0.0.1:8188"
POOL = ThreadPoolExecutor(max_workers=1)
LOCK = threading.RLock()
JOBS = {}
if DATA.exists():
    try:
        JOBS.update(json.loads(DATA.read_text(encoding="utf-8")))
    except Exception:
        pass
elif DRIVE_READY and (DRIVE / "jobs.json").exists():
    try:
        JOBS.update(json.loads((DRIVE / "jobs.json").read_text(encoding="utf-8")))
    except Exception:
        pass
SERVER = None

REQUIRED = [
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_video_vae_int8_convrot.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
]
H3_REPO = "Comfy-Org/MiniMax-H3"
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"


def cmd(args, cwd=None, stdout=None):
    p = subprocess.run([str(x) for x in args], cwd=str(cwd) if cwd else None,
                       text=True, stdout=stdout, stderr=subprocess.STDOUT if stdout else None)
    if p.returncode:
        raise RuntimeError("安裝或媒體處理失敗：" + " ".join(map(str, args[:5])))
    return p


def _save():
    with LOCK:
        content = json.dumps(JOBS, ensure_ascii=False, indent=2)
        DATA.write_text(content, encoding="utf-8")
        if DRIVE_READY:
            try:
                shutil.copy2(DATA, DRIVE / "jobs.json")
            except Exception:
                pass


def set_job(jobid, status, result=None):
    with LOCK:
        job = JOBS[jobid]
        job["status"] = str(status)
        job["updated"] = datetime.now().isoformat(timespec="seconds")
        if result:
            job["result"] = str(result)
        _save()


def model_state():
    if not (BASE / "main.py").exists():
        return "尚未安裝 ComfyUI"
    missing = [s for s in REQUIRED if not (BASE / "models" / s).exists()]
    return "✅ H3 四個模型已下載" if not missing else f"需要下載 {len(missing)} 個 H3 模型"


def gpu_state():
    try:
        import torch
        if torch.cuda.is_available():
            card = torch.cuda.get_device_properties(0)
            return f"{card.name}｜{card.total_memory/(1024**3):.1f} GiB 顯存"
    except Exception as exc:
        return f"GPU 偵測失敗：{exc}"
    return "未偵測到 GPU"


def ensure_comfy(jobid):
    if not (BASE / "main.py").exists():
        set_job(jobid, "安裝 ComfyUI（第一次執行）")
        cmd(["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git", BASE])
        cmd([sys.executable, "-m", "pip", "install", "-q", "-r", BASE / "requirements.txt"])
    for d in ["diffusion_models", "text_encoders", "vae", "upscale_models", "loras"]:
        (BASE / "models" / d).mkdir(parents=True, exist_ok=True)
    (BASE / "input").mkdir(exist_ok=True)
    (BASE / "output").mkdir(exist_ok=True)


def ensure_h3(jobid):
    ensure_comfy(jobid)
    from huggingface_hub import hf_hub_download
    for index, fname in enumerate(REQUIRED, start=1):
        path = BASE / "models" / fname
        if path.exists() and path.stat().st_size > 1000000:
            continue
        set_job(jobid, f"下載 H3 模型 {index}/4：{Path(fname).name}（首次約 37GB）")
        hf_hub_download(repo_id=H3_REPO, filename=fname, local_dir=str(BASE / "models"))
        if not path.exists() or path.stat().st_size < 1000000:
            raise RuntimeError("模型下載不完整：" + fname)
    set_job(jobid, "H3 模型就緒，準備啟動推論")


def ready():
    try:
        return requests.get(API + "/system_stats", timeout=3).status_code == 200
    except Exception:
        return False


def start_comfy(jobid):
    global SERVER
    if ready():
        return
    set_job(jobid, "啟動本機 ComfyUI API（不開公開 8188 網頁）")
    logfile = open(LOG, "a", buffering=1)
    SERVER = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188",
         "--lowvram", "--disable-pinned-memory"],
        cwd=str(BASE), stdout=logfile, stderr=subprocess.STDOUT)
    for _ in range(120):
        if ready():
            return
        if SERVER.poll() is not None:
            raise RuntimeError("ComfyUI 無法啟動；請查看 " + str(LOG))
        time.sleep(2)
    raise RuntimeError("ComfyUI API 啟動逾時，請查看 " + str(LOG))


def stop_comfy():
    global SERVER
    if SERVER is not None and SERVER.poll() is None:
        SERVER.terminate()
        try:
            SERVER.wait(timeout=20)
        except subprocess.TimeoutExpired:
            SERVER.kill()
        SERVER = None
    gc.collect()


def run_graph(graph, jobid, phase, timeout=10800):
    if not ready():
        raise RuntimeError("ComfyUI API 未啟動")
    response = requests.post(API + "/prompt", json={"prompt": graph}, timeout=90)
    if response.status_code != 200:
        raise RuntimeError("ComfyUI 節點驗證失敗：" + response.text[:1600])
    prompt_id = response.json()["prompt_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(8)
        h = requests.get(API + "/history/" + prompt_id, timeout=40).json()
        if prompt_id in h:
            item = h[prompt_id]
            status = item.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError("ComfyUI 模型錯誤：" + json.dumps(status, ensure_ascii=False)[-2000:])
            if status.get("completed") or status.get("status_str") == "success":
                return item
        set_job(jobid, f"{phase}：已執行 {int(timeout - (deadline-time.monotonic()))//60} 分鐘")
    raise TimeoutError(phase + " 超時")


def find_video(history, prefix=None):
    outputs = history.get("outputs", {})
    for item in outputs.values():
        if not isinstance(item, dict):
            continue
        for collection in item.values():
            if not isinstance(collection, list):
                continue
            for f in collection:
                if not isinstance(f, dict):
                    continue
                name = f.get("filename", "")
                if name.lower().endswith((".mp4", ".webm")):
                    p = BASE / f.get("type", "output") / f.get("subfolder", "") / name
                    if p.exists():
                        return p
    hits = sorted((BASE / "output").rglob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
    if prefix:
        hits = [x for x in hits if prefix in str(x)]
    if hits:
        return hits[0]
    raise RuntimeError("推論完成，但找不到影片輸出。")


def encode_mp4(input_path, output_path, start=None, duration=None):
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        args += ["-ss", str(start)]
    args += ["-i", str(input_path)]
    if duration:
        args += ["-t", str(duration)]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(output_path)]
    cmd(args)
    return output_path


def publish(jobid, video):
    # All public files stay under a dedicated, intentionally exposed results folder.
    final = PUBLIC / (jobid + ".mp4")
    encode_mp4(video, final)
    if DRIVE_READY:
        shutil.copy2(final, DRIVE / final.name)
    set_job(jobid, "✅ 完成｜可以從手機預覽與下載", final)
    return final


def h3_graph(image_name, prompt, w, h, length, seed, jobid):
    def n(cls, **kw):
        return {"class_type": cls, "inputs": kw}
    prefix = "H3Mobile/" + jobid
    return {
        "6": n("UNETLoader", unet_name=Path(REQUIRED[0]).name, weight_dtype="default"),
        "13": n("CLIPLoader", clip_name=Path(REQUIRED[1]).name, type="minimax", device="default"),
        "11": n("VAELoader", vae_name=Path(REQUIRED[2]).name),
        "24": n("VAELoader", vae_name=Path(REQUIRED[3]).name),
        "30": n("LoadImage", image=image_name),
        "104": n("MiniMaxH3ImageToVideo", clip=["13", 0], vae=["11", 0],
                 first_frame=["30", 0], width=w, height=h, length=length, prompt=prompt),
        "16": n("BasicGuider", model=["6", 0], conditioning=["104", 0]),
        "17": n("KSamplerSelect", sampler_name="res_multistep"),
        "9": n("BasicScheduler", model=["6", 0], scheduler="simple", steps=20, denoise=1),
        "15": n("RandomNoise", noise_seed=int(seed)),
        "14": n("SamplerCustomAdvanced", noise=["15", 0], guider=["16", 0],
                sampler=["17", 0], sigmas=["9", 0], latent_image=["104", 1]),
        "10": n("VAEDecode", samples=["14", 0], vae=["11", 0]),
        "23": n("VAEDecodeAudio", samples=["14", 0], vae=["24", 0]),
        "91": n("CreateVideo", images=["10", 0], audio=["23", 0], fps=24),
        "92": n("SaveVideo", video=["91", 0], filename_prefix=prefix,
                format="auto", codec="auto")
    }


def h3_worker(jobid, image_file, prompt, aspect, seconds, seed):
    try:
        ensure_h3(jobid)
        start_comfy(jobid)
        sizes = {"直式 9:16（測試）": (448, 800),
                 "直式 9:16（高清）": (736, 1312),
                 "橫式 16:9（測試）": (800, 448)}
        w, h = sizes[aspect]
        length = {5: 124, 10: 243, 15: 362}[int(seconds)]
        image_name = "h3_" + jobid + ".png"
        # Fit the first frame exactly, avoiding the H3 native aspect-ratio stretching.
        im = Image.open(image_file).convert("RGB")
        im = ImageOps.fit(im, (w, h), method=Image.Resampling.LANCZOS)
        im.save(BASE / "input" / image_name)
        set_job(jobid, f"H3 生成中：{w}×{h}，{seconds} 秒，原生音效")
        history = run_graph(h3_graph(image_name, prompt, w, h, length, seed, jobid),
                            jobid, "H3 正在生成", timeout=3*60*60)
        video = find_video(history, jobid)
        set_job(jobid, "轉換手機可播放的 MP4 並備份")
        publish(jobid, video)
    except Exception as exc:
        set_job(jobid, "❌ " + str(exc)[:900])


def ffprobe(path):
    return json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        text=True))


def ffmpeg_frames(video, frame_dir, limit):
    frame_dir.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video)]
    if limit:
        command += ["-t", str(limit)]
    command += ["-vsync", "0", str(frame_dir / "%06d.png")]
    cmd(command)


def upscale_frames(jobid, video, output, width, height, test_seconds):
    ensure_comfy(jobid)
    model = BASE / "models/upscale_models/RealESRGAN_x4plus_anime_6B.pth"
    if not model.exists():
        set_job(jobid, "下載 Real-ESRGAN Anime 權重")
        urllib.request.urlretrieve(ESRGAN_URL, model)
    if model.stat().st_size < 1_000_000:
        raise RuntimeError("Real-ESRGAN 權重下載不完整")
    start_comfy(jobid)
    try:
        requests.post(API+"/free",json={"unload_models":True,"free_memory":True},timeout=20)
    except Exception:
        pass
    meta = ffprobe(video)
    stream = next(x for x in meta["streams"] if x["codec_type"] == "video")
    fps = float(Fraction(stream.get("avg_frame_rate", "24/1")))
    duration = float(meta["format"]["duration"])
    cliplen = min(duration, float(test_seconds)) if test_seconds else duration
    raw = output.parent / "input_frames"
    frames = output.parent / "enhanced_frames"
    raw.mkdir(exist_ok=True)
    frames.mkdir(exist_ok=True)
    ffmpeg_frames(video, raw, cliplen)
    parts = sorted(raw.glob("*.png"))
    for i, p in enumerate(parts):
        image_name = jobid + "_" + p.name
        shutil.copy2(p, BASE / "input" / image_name)
        g = {
            "1": {"class_type":"LoadImage", "inputs":{"image":image_name}},
            "2": {"class_type":"UpscaleModelLoader","inputs":{"model_name":model.name}},
            "3": {"class_type":"ImageUpscaleWithModel","inputs":{"upscale_model":["2",0],"image":["1",0]}},
            "4": {"class_type":"ImageScale","inputs":{
                  "image":["3",0],"upscale_method":"lanczos",
                  "width":int(width),"height":int(height),"crop":"center"}},
            "5": {"class_type":"SaveImage","inputs":{
                  "images":["4",0],"filename_prefix":"H3Mobile/upscale_" + jobid}}
        }
        hist = run_graph(g, jobid, "AI 動漫逐幀放大", timeout=1800)
        f = hist.get("outputs",{}).get("5",{}).get("images",[])
        if not f:
            raise RuntimeError("畫質放大沒有返回影格")
        item = f[0]
        shutil.copy2(BASE / "output" / item.get("subfolder","") / item["filename"], frames / p.name)
        if (i+1) % 10 == 0 or i+1 == len(parts):
            set_job(jobid, f"Real-ESRGAN 動漫畫質提升：{i+1}/{len(parts)} 幀")
    audio = any(s["codec_type"]=="audio" for s in meta["streams"])
    args = ["ffmpeg","-hide_banner","-loglevel","error","-y",
            "-framerate",str(fps),"-i",str(frames/"%06d.png"),"-i",str(video),
            "-map","0:v:0"]
    if audio:
        args+=["-map","1:a:0"]
    args+=["-t",str(cliplen),"-c:v","libx264","-crf","18","-pix_fmt","yuv420p"]
    if audio:
        args+=["-c:a","aac"]
    args.append(str(output))
    cmd(args)
    return cliplen


def synth_voice(jobid, text, speaker, style, output):
    set_job(jobid, "載入 Qwen3-TTS 中文配音（首次需下載模型）")
    try:
        import qwen_tts
    except ImportError:
        set_job(jobid, "首次安裝 Qwen3-TTS 套件")
        cmd([sys.executable, "-m", "pip", "install", "-q", "qwen-tts", "soundfile"])
    # H3 runs in a separate process; terminate it to free L4 GPU before loading TTS.
    stop_comfy()
    import torch
    from qwen_tts import Qwen3TTSModel
    import soundfile as sf
    tts = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa")
    audio, sr = tts.generate_custom_voice(
        text=text, language="Chinese", speaker=speaker, instruct=style)
    sf.write(str(output), audio[0], sr)
    del tts
    gc.collect()
    torch.cuda.empty_cache()
    return output


def post_worker(jobid, source, upscale, quality, test_seconds, speech, speaker, style, music_file, subtitles, interpolate):
    try:
        job_dir = WORK/jobid
        clip = job_dir/"input.mp4"
        meta = ffprobe(clip)
        duration = float(meta.get("format",{}).get("duration",0))
        if duration<=0:
            raise RuntimeError("無法讀取影片時長")
        limit = min(duration, float(test_seconds)) if test_seconds else duration
        if upscale:
            width,height = (720,1280) if quality=="720p（推薦 L4）" else (1080,1920)
            current=job_dir/"upscaled.mp4"
            set_job(jobid, "準備 Real-ESRGAN AI 畫質增強")
            upscale_frames(jobid, clip, current, width,height,limit)
        else:
            current=clip
        voice=job_dir/"voice_zh.wav"
        if speech.strip():
            synth_voice(jobid, speech.strip(),speaker,style,voice)
        set_job(jobid, "合成音樂、原生音效、中文配音與字幕")
        meta2=ffprobe(current)
        aud=any(x["codec_type"]=="audio" for x in meta2["streams"])
        args=["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(current)]
        has_voice=voice.exists()
        if has_voice:
            args+=["-i",str(voice)]
        music=Path(music_file) if music_file else None
        if music and music.exists():
            args+=["-stream_loop","-1","-i",str(music)]
        has_music=music is not None and music.exists()
        streams=[]
        if aud:
            streams+=["[0:a:0]volume=0.25[a0]"]
        if has_voice:
            streams+=["[1:a:0]volume=1.2[a1]"]
        if has_music:
            mi=2 if has_voice else 1
            streams += [f"[{mi}:a:0]volume=0.15[a2]"]
        labels=[x for x,t in [("a0",aud),("a1",has_voice),("a2",has_music)] if t]
        if labels:
            if len(labels)==1:
                audio_map=f"[{labels[0]}]"
            else:
                streams += ["".join(f"[{x}]" for x in labels)+f"amix=inputs={len(labels)}:duration=first:dropout_transition=0[amix]"]
                audio_map="[amix]"
            args += ["-filter_complex",";".join(streams)]
        else:
            audio_map=None
        # Text subtitle is based on supplied dialogue. It is not forced-alignment.
        if subtitles and has_voice:
            def stamp(n):
                millis=round(max(n,0)*1000)
                hours,rem=divmod(millis,3600000)
                mins,rem=divmod(rem,60000)
                secs,ms=divmod(rem,1000)
                return f"{hours:02d}:{mins:02d}:{secs:02d},{ms:03d}"
            voicedur=float(ffprobe(voice).get("format",{}).get("duration",limit))
            subtitle_file=job_dir/"subtitle.srt"
            subtitle_file.write_text(f"1\n00:00:00,000 --> {stamp(min(limit,voicedur))}\n{speech.strip()}\n",encoding="utf-8")
            args+=["-vf",f"subtitles={subtitle_file}:force_style='FontName=Noto Sans CJK TC,Fontsize=18,Outline=2,Alignment=2,MarginV=65'"]
        if interpolate:
            # ffmpeg minterpolate is optical flow interpolation, NOT RIFE.
            # Compose with subtitle video filter if needed.
            vf = f"minterpolate=fps=48:mi_mode=mci"
            if subtitles and has_voice:
                args[-1] = args[-1]+","+vf
            else:
                args+=["-vf",vf]
        args+=["-map","0:v:0"]
        if audio_map:
            args+=["-map",audio_map]
        args+=["-t",str(limit),"-c:v","libx264","-crf","18",
               "-preset","veryfast","-pix_fmt","yuv420p"]
        if audio_map:
            args+=["-c:a","aac","-b:a","192k"]
        output=job_dir/"final.mp4"
        args+=["-movflags","+faststart",str(output)]
        cmd(args)
        if subtitles and has_voice:
            shutil.copy2(subtitle_file, PUBLIC/(jobid+".srt"))
            if DRIVE_READY:
                shutil.copy2(subtitle_file,DRIVE/(jobid+".srt"))
        publish(jobid,output)
    except Exception as exc:
        set_job(jobid,"❌ "+str(exc)[:900])


def make_job(kind, source=None):
    jobid=datetime.now().strftime("%m%d-%H%M")+"-"+uuid.uuid4().hex[:6]
    (WORK/jobid).mkdir(exist_ok=True)
    with LOCK:
        JOBS[jobid]={"id":jobid,"kind":kind,"status":"排隊中","updated":datetime.now().isoformat(timespec="seconds"),"result":None}
        _save()
    return jobid


def job_options():
    with LOCK:
        return [(f"{v['kind']}｜{v['id']}｜{v['status'][:35]}",k)
                for k,v in reversed(list(JOBS.items()))]


def submit_h3(image, prompt, aspect, seconds, seed):
    if not image:
        raise gr.Error("請先上傳一張角色圖片")
    if not prompt.strip():
        raise gr.Error("請輸入動作提示詞")
    jobid=make_job("H3 圖生影片")
    local=WORK/jobid/"image.png"
    shutil.copy2(image,local)
    POOL.submit(h3_worker,jobid,str(local),prompt,aspect,seconds,int(seed))
    return gr.update(choices=job_options(),value=jobid),f"✅ 已提交任務 {jobid}，可以關閉手機畫面；只要 Colab 仍運作，後端會繼續處理。"


def submit_post(video, previous_job, upscale, quality, test_seconds, speech, speaker, style, music, subtitles, interpolate):
    if not video and not previous_job:
        raise gr.Error("請上傳影片，或選擇一個已完成的 H3 任務")
    jobid=make_job("AI 畫質／配音")
    dest=WORK/jobid/"input.mp4"
    if video:
        shutil.copy2(video,dest)
    else:
        with LOCK:
            record=JOBS.get(previous_job)
            source=Path(record.get("result","")) if record and record.get("result") else None
        if not source or not source.is_file():
            raise gr.Error("選取的任務尚未完成或影片已不存在")
        shutil.copy2(source,dest)
    track=None
    if music:
        track=WORK/jobid/("music"+Path(music).suffix)
        shutil.copy2(music,track)
    POOL.submit(post_worker,jobid,str(dest),bool(upscale),quality,float(test_seconds),
                speech,speaker,style,str(track) if track else None,bool(subtitles),bool(interpolate))
    return gr.update(choices=job_options(),value=jobid),f"✅ 後製任務 {jobid} 已加入佇列。"


def check_job(jobid):
    if not jobid:
        return "選擇任務查看進度。",None,None
    with LOCK:
        record=JOBS.get(jobid,{})
    message=f"**任務：{jobid}**\n\n{record.get('status','找不到任務')}\n\n最後更新：{record.get('updated','—')}"
    path=Path(record["result"]) if record.get("result") else None
    if (not path or not path.is_file()) and DRIVE_READY:
        backup=DRIVE/(jobid+".mp4")
        if backup.is_file():
            path=backup
    if path and path.is_file():
        return message,str(path),str(path)
    return message,None,None


def refresh_jobs():
    choices=job_options()
    value=choices[0][1] if choices else None
    return gr.update(choices=choices,value=value), gr.update(choices=choices,value=None)


CSS="""
.gradio-container { max-width: 920px !important; margin: auto; }
#hero { padding:20px; border-radius:20px; background:linear-gradient(135deg,#16253d,#263d64); color:#fff; }
#hero h1 { color:#fff !important; margin:0; }
button.primary { min-height:52px; font-size:17px !important; border-radius:14px !important; }
"""

def app():
    with gr.Blocks(title="H3 Mobile Studio",css=CSS,analytics_enabled=False) as demo:
        gr.HTML('<div id="hero"><h1>🎬 H3 Mobile Studio</h1><p>手機上傳・Colab GPU 推論・繁中配音・畫質增強・雲端成品</p></div>')
        gr.Markdown("**使用方式：** 先送出任務，再到「任務／下載」查看進度。手機可以離開本頁，但 Colab 執行階段必須保持運作。首次 H3 生成會自動下載約 37GB 模型。")
        with gr.Tabs():
            with gr.Tab("🎞 H3 圖生影片"):
                image=gr.Image(label="上傳動漫角色／首幀（PNG、JPG）",type="filepath")
                prompt=gr.Textbox(label="動畫提示詞（可使用中英文）",lines=5,value=(
                    "Preserve the exact anime character identity, hair, face, and outfit. "
                    "The character draws a glowing sword and makes one cinematic diagonal slash. "
                    "Detailed Japanese anime cel-shading, smooth movement and camera tracking. "
                    "Audio: wind and sword sound effects, no dialogue. No subtitles, no text."))
                with gr.Row():
                    aspect=gr.Dropdown(choices=["直式 9:16（測試）","直式 9:16（高清）","橫式 16:9（測試）"],
                                       value="直式 9:16（測試）",label="畫面尺寸")
                    secs=gr.Dropdown(choices=[5,10,15],value=5,label="片長（秒）")
                seed=gr.Number(value=42,precision=0,label="隨機種子")
                make=gr.Button("🚀 提交 H3 生成任務",variant="primary")
                gr.Markdown("提示：L4 24GB 建議先用 5 秒測試；高清/15 秒可能顯存不足。H3 包含原生音效。")
            with gr.Tab("✨ 畫質＋中文配音"):
                v_in=gr.Video(label="上傳要後製的 MP4（若有上傳，優先使用此影片）")
                completed=gr.Dropdown(choices=job_options(),label="或選擇已完成的 H3 任務",value=None)
                with gr.Row():
                    upscale=gr.Checkbox(value=True,label="Real-ESRGAN 動漫 AI 畫質增強")
                    quality=gr.Dropdown(choices=["720p（推薦 L4）","1080p（高顯存）"],
                                        value="720p（推薦 L4）",label="輸出尺寸（直式）")
                test=gr.Dropdown(choices=[5,10,15,0],value=5,
                                 label="先處理前幾秒？0 = 整段")
                speech=gr.Textbox(label="中文配音文字（留空則不生成配音）",lines=3,
                                   value="吾以天命之名，斬破這無盡的黑暗！")
                speaker=gr.Dropdown(choices=["Uncle_Fu","Vivian","Serena","Ryan"],
                                    value="Uncle_Fu",label="Qwen3-TTS 聲線")
                style=gr.Textbox(label="中文情緒描述",value="用有力、堅定的語氣說，語速偏慢。")
                music=gr.Audio(label="可選：上傳背景音樂",type="filepath",sources=["upload"])
                with gr.Row():
                    subs=gr.Checkbox(value=True,label="將台詞顯示為繁體字幕")
                    interpolate=gr.Checkbox(value=False,label="FFmpeg 光流補幀至 48fps（非 RIFE）")
                edit=gr.Button("✨ 提交影音精修任務",variant="primary")
                gr.Markdown("畫質和中文配音分段執行，避免同時載入 H3、Real-ESRGAN 和 TTS 而耗盡 GPU。字幕以輸入台詞顯示，不是逐字對齊。")
            with gr.Tab("📥 任務／下載"):
                jobs=gr.Dropdown(label="選擇任務",choices=job_options(),value=(job_options()[0][1] if job_options() else None))
                refresh=gr.Button("🔄 更新任務清單")
                progress=gr.Markdown("任務執行進度會在此顯示")
                preview=gr.Video(label="完成影片預覽",format="mp4")
                download=gr.File(label="下載 MP4 成品")
                timer=gr.Timer(10,active=True)
            with gr.Tab("⚙️ GPU／系統"):
                g=gr.Markdown("**GPU：** "+gpu_state()+"\n\n**模型：** "+model_state()+
                              "\n\n**成品資料夾：** "+str(DRIVE if DRIVE_READY else PUBLIC))
                gr.Markdown("注意：Gradio share 網址是暫時的；登入後請勿轉給不認識的人。Colab Pro 不保證 24 小時常駐。需要 24/7 應改成固定的雲端 GPU 伺服器。")
        make.click(submit_h3,inputs=[image,prompt,aspect,secs,seed],outputs=[jobs,progress],show_progress="minimal")
        edit.click(submit_post,inputs=[v_in,completed,upscale,quality,test,speech,speaker,style,music,subs,interpolate],
                   outputs=[jobs,progress],show_progress="minimal")
        refresh.click(refresh_jobs,outputs=[jobs,completed],show_progress="hidden")
        jobs.change(check_job,inputs=jobs,outputs=[progress,preview,download],show_progress="hidden")
        timer.tick(check_job,inputs=jobs,outputs=[progress,preview,download],show_progress="hidden")
    demo.queue(default_concurrency_limit=4,max_size=20)
    return demo


def launch(username, password):
    if not password or len(password)<10:
        raise ValueError("為保護你的 GPU，請設定至少 10 個字元的登入密碼")
    demo=app()
    return demo.launch(share=True,pwa=True,auth=(username,password),inline=False,
                       server_name="0.0.0.0",server_port=7860,
                       show_error=True,prevent_thread_lock=False,allowed_paths=[str(PUBLIC)])
