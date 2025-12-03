# main.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import torch
import shutil
import os
import numpy as np

# 우리가 만든 모듈들 임포트
from app.model import resnet18_transfer_learning
from app.separator import separate_audio, mel_spectrogram # mel_spectrogram 추가
from app.search import VectorSearchEngine

app = FastAPI(title="Music Timbre Search AI Server")

# --- 전역 변수 및 설정 ---
MODEL = None
SEARCH_ENGINE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_WIDTH = 431 # 훈련 시 사용했던 스펙트로그램 너비

@app.on_event("startup")
async def load_resources():
    global MODEL, SEARCH_ENGINE
    
    # 1. 모델 로드
    print("Loading AI Model...")
    MODEL = resnet18_transfer_learning().to(DEVICE)
    
    # --- 수정: 모델 경로를 best_model.pth로 변경 ---
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        # 이제 모델이 없으면 치명적인 오류로 간주
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

@app.post("/api/search")
async def search_music(
    # --- 수정: start_sec, end_sec 제거 ---
    file: UploadFile = File(...), 
    instrument: str = Form(...)
):
    """
    사용자로부터 받은 오디오 클립(이미 구간이 잘린 상태)과 악기 정보를 받아 유사한 음악을 검색합니다.
    """
    temp_filename = f"temp_{file.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # --- 수정된 로직 ---
        # 1. 음원 분리 (demucs)
        stems, sr = separate_audio(temp_filename, device=DEVICE)
        if stems is None or instrument not in stems:
            raise HTTPException(status_code=400, detail=f"Could not separate the '{instrument}' track.")
        
        instrumental_track = stems[instrument]

        # --- 최종 수정: 스펙트로그램 생성 직전에 모노로 변환 ---
        # separate_audio는 스테레오([2, n])를 반환하므로, 여기서 모노로 변환합니다.
        mono_instrumental_track = torch.mean(instrumental_track, dim=0, keepdim=True)
        # --- 수정 끝 ---

        # 2. 스펙트로그램 변환
        long_spectrogram = mel_spectrogram(mono_instrumental_track, sample_rate=sr)
        if long_spectrogram is None:
            raise HTTPException(status_code=400, detail="Spectrogram conversion failed.")

        long_spectrogram = long_spectrogram.to(DEVICE)
        
        # 3. 슬라이딩 윈도우로 여러 조각(chunk) 생성
        total_frames = long_spectrogram.shape[-1]
        window_size = TARGET_WIDTH
        stride = window_size // 2  # 50% overlap

        chunks = []
        if total_frames <= window_size:
            # 입력이 5초보다 짧거나 같으면, 크기를 맞춰서 그대로 사용
            padded_chunk = torch.zeros(1, long_spectrogram.shape[1], window_size, device=DEVICE)
            padded_chunk[..., :total_frames] = long_spectrogram
            chunks.append(padded_chunk)
        else:
            # 입력이 5초보다 길면, 슬라이딩하며 조각 추출
            for i in range(0, total_frames - window_size + 1, stride):
                chunk = long_spectrogram[..., i:i + window_size]
                chunks.append(chunk)
        
        if not chunks:
             raise HTTPException(status_code=400, detail="Could not extract any valid audio chunks.")

        batch = torch.cat(chunks, dim=0)

        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        
        # --- GPU/CPU 장치 일치 ---
        batch = batch.to(DEVICE)

        # 4. 모델로 모든 조각을 한 번에 추론하고 평균내어 쿼리 벡터 생성
        with torch.no_grad():
            all_embeddings = MODEL(batch)
        query_vector = torch.mean(all_embeddings, dim=0, keepdim=True)
        # --- 로직 수정 끝 ---

        # 5. Faiss 검색 수행
        results = SEARCH_ENGINE.search(query_vector.cpu().numpy(), top_k=5)

        return {
            "status": "success",
            "results": results
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)