# main.py
from httpx import request
import uvicorn
from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
import torch
import shutil
import os
import numpy as np
import uuid
import yt_dlp # 유튜브 다운로드를 위해 추가
import torchaudio # 오디오 자르기를 위해 추가
import time

# 우리가 만든 모듈들 임포트
from app.model import resnet18_transfer_learning
from app.separator import separate_audio, mel_spectrogram
from app.search import VectorSearchEngine

class RecommendRequest(BaseModel):
    youtube_url: str
    instrument: str
    start_sec: float
    end_sec: float

app = FastAPI(title="Music Timbre Search AI Server")

# --- 전역 변수 및 설정 ---
MODEL = None
SEARCH_ENGINE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_WIDTH = 431 # 훈련 시 사용했던 스펙트로그램 너비
TEMP_DOWNLOAD_DIR = "temp_downloads" # 다운로드 및 임시 파일을 저장할 폴더
COOKIE_FILE = "cookies.txt" # 쿠키 파일 이름

@app.on_event("startup")
async def load_resources():
    global MODEL, SEARCH_ENGINE
    
    # 임시 폴더 생성
    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)

    # 1. 모델 로드
    print("Loading AI Model...")
    MODEL = resnet18_transfer_learning().to(DEVICE)
    
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        raise RuntimeError(f"FATAL: Model weights not found at {model_path}. Please run 'scripts/train.py' first.")
        
    MODEL.eval()
    
    # 2. 검색 엔진 로드
    print("Loading Search Engine...")
    index_path = "indexes/timbre.index"
    metadata_path = "indexes/metadata.json"
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        SEARCH_ENGINE = VectorSearchEngine(index_path, metadata_path)
    else:
        raise RuntimeError("FATAL: Index or metadata file not found. Please run 'scripts/build_index.py' first.")

@app.post("/recommend")
async def recommend_music(request: RecommendRequest):
    """
    유튜브 링크, 시간, 악기 정보를 받아 유사한 음악을 검색합니다.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    download_path = os.path.join(TEMP_DOWNLOAD_DIR, request_id)
    clipped_path = os.path.join(TEMP_DOWNLOAD_DIR, f"{request_id}_clipped.wav")

    try:
        # 1. 유튜브 오디오 다운로드
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] Downloading audio from: {request.youtube_url}")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': download_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'noplaylist': True,
        }
        
        # cookies.txt 파일이 있으면 옵션에 추가
        if os.path.exists(COOKIE_FILE):
            print(f"Using cookie file: {COOKIE_FILE}")
            ydl_opts['cookiefile'] = COOKIE_FILE
        else:
            print("Cookie file not found, proceeding without it.")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(request.youtube_url, download=True)
            video_title = info_dict.get('title', None)
            if not video_title:
                raise HTTPException(status_code=500, detail="Could not extract video title.")
        
        # yt-dlp가 확장자를 .wav로 바꿔주므로 실제 파일 경로를 다시 확인
        actual_download_path = os.path.join(TEMP_DOWNLOAD_DIR, f"{request_id}.wav")
        if not os.path.exists(actual_download_path):
             raise HTTPException(status_code=500, detail="Downloaded audio file not found.")

        # 2. 오디오 파일 자르기
        print(f"Clipping audio from {request.start_sec}s to {request.end_sec}s")
        waveform, sr = torchaudio.load(actual_download_path)
        start_frame = int(request.start_sec * sr)
        end_frame = int(request.end_sec * sr)
        clipped_waveform = waveform[:, start_frame:end_frame]
        
        if clipped_waveform.shape[1] == 0:
            raise HTTPException(status_code=400, detail="The specified time range is invalid or contains no audio.")
            
        torchaudio.save(clipped_path, clipped_waveform, sr)

        # 3. 음원 분리 (demucs)
        print(f"Separating '{request.instrument}' track...")
        stems, sr_sep = separate_audio(clipped_path, device=DEVICE)
        if stems is None or request.instrument not in stems:
            raise HTTPException(status_code=400, detail=f"Could not separate the '{request.instrument}' track.")
        
        instrumental_track = stems[request.instrument]

        # 4. 모노 변환 및 스펙트로그램 생성
        mono_instrumental_track = torch.mean(instrumental_track, dim=0, keepdim=True)
        long_spectrogram = mel_spectrogram(mono_instrumental_track, sample_rate=sr_sep)
        if long_spectrogram is None:
            raise HTTPException(status_code=400, detail="Spectrogram conversion failed.")

        long_spectrogram = long_spectrogram.to(DEVICE)
        
        # 5. 슬라이딩 윈도우로 쿼리 벡터 생성
        total_frames = long_spectrogram.shape[-1]
        window_size = TARGET_WIDTH
        stride = window_size // 2

        chunks = []
        if total_frames <= window_size:
            padded_chunk = torch.zeros(1, long_spectrogram.shape[1], window_size, device=DEVICE)
            padded_chunk[..., :total_frames] = long_spectrogram
            chunks.append(padded_chunk)
        else:
            for i in range(0, total_frames - window_size + 1, stride):
                chunk = long_spectrogram[..., i:i + window_size]
                chunks.append(chunk)
        
        if not chunks:
             raise HTTPException(status_code=400, detail="Could not extract any valid audio chunks.")

        batch = torch.cat(chunks, dim=0)
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        
        batch = batch.to(DEVICE)

        # 6. 모델 추론 및 쿼리 벡터 생성
        with torch.no_grad():
            all_embeddings = MODEL(batch)
        query_vector = torch.mean(all_embeddings, dim=0, keepdim=True)

        # 7. Faiss 검색 수행
        print(f"Searching for similar tracks, excluding '{video_title}'...")
        results = SEARCH_ENGINE.search(
            query_vector.cpu().numpy(), 
            top_k=5,
            exclude_song_name=video_title
        )
        end_time = time.time()
        print(f"[DEBUG]Found {len(results)} similar tracks.")
        print(f"[DEBUG]Search took {end_time - start_time:.2f} seconds.")
        print("--- Results ---")
        print("[DEBUG]", results)

        return {
            "status": "success",
            "results": results
        }

    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 작업 완료 후 모든 임시 파일 삭제
        if 'actual_download_path' in locals() and os.path.exists(actual_download_path):
            os.remove(actual_download_path)
        if os.path.exists(clipped_path):
            os.remove(clipped_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)