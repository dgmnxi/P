# main.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import torch
import shutil
import os
import numpy as np
import uuid

# 우리가 만든 모듈들 임포트
from app.model import resnet18_transfer_learning
from app.separator import separate_audio, mel_spectrogram
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
async def recommend_music(
    file: UploadFile = File(...), 
    instrument: str = Form(...)
):
    """
    사용자로부터 받은 오디오 클립(이미 구간이 잘린 상태)과 악기 정보를 받아 유사한 음악을 검색합니다.
    """
    # 각 요청마다 고유한 임시 파일 이름 생성
    temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
    
    try:
        # 1. 업로드된 파일을 임시 저장
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. 음원 분리 (demucs)
        stems, sr = separate_audio(temp_filename, device=DEVICE)
        if stems is None or instrument not in stems:
            raise HTTPException(status_code=400, detail=f"Could not separate the '{instrument}' track.")
        
        instrumental_track = stems[instrument]

        # 3. 모노 변환 및 스펙트로그램 생성
        mono_instrumental_track = torch.mean(instrumental_track, dim=0, keepdim=True)
        long_spectrogram = mel_spectrogram(mono_instrumental_track, sample_rate=sr)
        if long_spectrogram is None:
            raise HTTPException(status_code=400, detail="Spectrogram conversion failed.")

        long_spectrogram = long_spectrogram.to(DEVICE)
        
        # 4. 슬라이딩 윈도우로 여러 조각(chunk) 생성
        total_frames = long_spectrogram.shape[-1]
        window_size = TARGET_WIDTH
        stride = window_size // 2  # 50% overlap

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

        # 5. 모델로 모든 조각을 추론하고 평균내어 쿼리 벡터 생성
        with torch.no_grad():
            all_embeddings = MODEL(batch)
        query_vector = torch.mean(all_embeddings, dim=0, keepdim=True)

        # 6. Faiss 검색 수행
        results = SEARCH_ENGINE.search(query_vector.cpu().numpy(), top_k=5)

        return {
            "status": "success",
            "results": results
        }

    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 작업 완료 후 임시 업로드 파일 삭제
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)