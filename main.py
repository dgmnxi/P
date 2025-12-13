# main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import Union
import torch
import shutil
import os
import numpy as np
import uuid
import yt_dlp
import torchaudio
import time
import re # 제목 정제를 위해 re 모듈 추가

# 우리가 만든 모듈들 임포트
from app.model import resnet18_transfer_learning
from app.separator import separate_audio, mel_spectrogram
from app.search import VectorSearchEngine

class RecommendRequest(BaseModel):
    youtube_url: str
    instrument: list[str]
    start_sec: float
    end_sec: float
    top_k: int = 5

    @validator('instrument', pre=True)
    def parse_instrument(cls, v):
        # 입력이 문자열이면 리스트로 감싸서 반환
        if isinstance(v, str):
            return [v]
        return v

app = FastAPI(title="Music Timbre Search AI Server")

# --- 전역 변수 및 설정 ---
MODEL = None
SEARCH_ENGINE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_WIDTH = 431
TEMP_DOWNLOAD_DIR = "temp_downloads"
COOKIE_FILE = "cookies.txt"

@app.on_event("startup")
async def load_resources():
    global MODEL, SEARCH_ENGINE
    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
    print("Loading AI Model...")
    MODEL = resnet18_transfer_learning().to(DEVICE)
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        raise RuntimeError(f"FATAL: Model weights not found at {model_path}.")
    MODEL.eval()
    print("Loading Search Engine...")
    index_path = "indexes/timbre.index"
    metadata_path = "indexes/metadata.json"
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        SEARCH_ENGINE = VectorSearchEngine(index_path, metadata_path)
    else:
        raise RuntimeError("FATAL: Index or metadata file not found.")

@app.post("/recommend")
async def recommend_music(request: RecommendRequest):

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
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            'noplaylist': True,
        }
        if os.path.exists(COOKIE_FILE):
            ydl_opts['cookiefile'] = COOKIE_FILE

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(request.youtube_url, download=True)
            video_title = info_dict.get('title', 'Unknown')

        # --- 제목 정제 로직 추가 ---(build_index.py와 동일한 방식으로 제목 정제)
        cleaned_title = video_title
        if ' - ' in video_title:
            try:
                # 아티스트 부분은 버리고 제목 부분만 사용
                _, title_part = video_title.split(' - ', 1)
                cleaned_title = title_part.strip()
            except ValueError:
                pass
        
        cleaned_title = re.sub(r'\s*\(.*?\)|\[.*?\]', '', cleaned_title).strip()
        cleaned_title = re.sub(r'(?i)\s*(ft|feat|m/v|official|video|audio|lyric|lyrics)\s*.*', '', cleaned_title).strip()
        if cleaned_title.startswith('"') and cleaned_title.endswith('"'):
            cleaned_title = cleaned_title[1:-1]
        if cleaned_title.startswith("'") and cleaned_title.endswith("'"):
            cleaned_title = cleaned_title[1:-1]
        # --- 로직 끝 ---

        actual_download_path = os.path.join(TEMP_DOWNLOAD_DIR, f"{request_id}.wav")
        if not os.path.exists(actual_download_path):
             raise HTTPException(status_code=500, detail="Downloaded audio file not found.")

        # 2. 오디오 파일 자르기
        waveform, sr = torchaudio.load(actual_download_path)
        start_frame = int(request.start_sec * sr)
        end_frame = int(request.end_sec * sr)
        clipped_waveform = waveform[:, start_frame:end_frame]
        if clipped_waveform.shape[1] == 0:
            raise HTTPException(status_code=400, detail="The specified time range is invalid or contains no audio.")
        torchaudio.save(clipped_path, clipped_waveform, sr)

        # 3. 음원 분리
        stems, sr_sep = separate_audio(clipped_path, device=DEVICE)
        if stems is None:
            raise HTTPException(status_code=400, detail="Could not separate audio tracks.")

        all_results = []
        # 요청된 모든 악기에 대해 검색 수행
        for instrument_name in request.instrument:
            if instrument_name not in stems:
                print(f"Warning: '{instrument_name}' track not found. Skipping.")
                continue

            instrumental_track = stems[instrument_name]

            # 4. 스펙트로그램 생성
            mono_instrumental_track = torch.mean(instrumental_track, dim=0, keepdim=True)
            long_spectrogram = mel_spectrogram(mono_instrumental_track, sample_rate=sr_sep)
            if long_spectrogram is None:
                print(f"Warning: Spectrogram conversion failed for '{instrument_name}'. Skipping.")
                continue
            long_spectrogram = long_spectrogram.to(DEVICE)
            
            # 5. 쿼리 벡터 생성
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
                    chunks.append(long_spectrogram[..., i:i + window_size])
            
            if not chunks:
                print(f"Warning: Could not extract valid chunks for '{instrument_name}'. Skipping.")
                continue

            batch = torch.cat(chunks, dim=0).unsqueeze(1)
            with torch.no_grad():
                all_embeddings = MODEL(batch)
            query_vector = torch.mean(all_embeddings, dim=0, keepdim=True)

            # 7. Faiss 검색 수행
            print(f"Searching for instrument '{instrument_name}', excluding '{cleaned_title}'...")
            # 검색 시 악기 필터를 사용하여 해당 악기만 검색하도록 함
            results = SEARCH_ENGINE.search(
                query_vector.cpu().numpy(), 
                top_k=request.top_k * 3, # 각 악기별로 더 많은 후보를 가져옴
                exclude_title=cleaned_title,
                instruments=[instrument_name] # 악기 필터링
            )
            all_results.extend(results)

        # 모든 악기 검색 결과를 종합하여 유사도 순으로 정렬
        all_results.sort(key=lambda x: x['similarity'], reverse=True)

        # 최종적으로 중복 없는 top_k개의 결과를 선택
        final_results = []
        seen_titles = set()
        for res in all_results:
            if len(final_results) >= request.top_k:
                break
            if res['title'] not in seen_titles:
                final_results.append(res)
                seen_titles.add(res['title'])

        end_time = time.time()
        print(f"Search took {end_time - start_time:.2f} seconds. Found {len(final_results)} unique results.")
        #-------DEBUG----------
        print("Final Results:", final_results)

        return {"status": "success", "results": final_results}

    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if 'actual_download_path' in locals() and os.path.exists(actual_download_path):
            os.remove(actual_download_path)
        if os.path.exists(clipped_path):
            os.remove(clipped_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)