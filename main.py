# main.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import torch
import shutil
import os
import numpy as np
import uuid # 고유 ID 생성을 위해 추가

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
TEMP_SEPARATED_DIR = "temp_separated" # 분리된 음원을 저장할 임시 폴더

@app.on_event("startup")
async def load_resources():
    global MODEL, SEARCH_ENGINE
    
    # 임시 폴더 생성
    os.makedirs(TEMP_SEPARATED_DIR, exist_ok=True)

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

@app.post("/separate")
async def separate_track(
    file: UploadFile = File(...), 
    instrument: str = Form(...)
):
    """
    오디오 파일을 받아 특정 악기 트랙을 분리하고,
    처리된 트랙에 대한 고유 ID를 반환합니다.
    """
    temp_upload_filename = f"temp_upload_{uuid.uuid4()}"
    
    try:
        with open(temp_upload_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. 음원 분리 (demucs)
        stems, sr = separate_audio(temp_upload_filename, device=DEVICE)
        if stems is None or instrument not in stems:
            raise HTTPException(status_code=400, detail=f"Could not separate the '{instrument}' track.")
        
        instrumental_track = stems[instrument]

        # 2. 분리된 트랙을 고유 ID와 함께 임시 저장
        separated_id = str(uuid.uuid4())
        separated_filepath = os.path.join(TEMP_SEPARATED_DIR, f"{separated_id}.pt")
        
        torch.save({
            'track': instrumental_track.cpu(), # CPU에 저장하여 호환성 확보
            'sr': sr
        }, separated_filepath)

        return {
            "status": "success",
            "message": "Separation complete.",
            "separated_id": separated_id
        }

    except Exception as e:
        print(f"Error during separation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_upload_filename):
            os.remove(temp_upload_filename)

@app.post("/recommend")
async def recommend_music(
    separated_id: str = Form(...)
):
    """
    분리된 트랙의 ID를 받아, 해당 트랙으로 유사 음악을 검색합니다.
    """
    separated_filepath = os.path.join(TEMP_SEPARATED_DIR, f"{separated_id}.pt")
    
    if not os.path.exists(separated_filepath):
        raise HTTPException(status_code=404, detail="Separated track not found. Please run /api/separate first.")

    try:
        # 1. 저장된 트랙 데이터 로드
        data = torch.load(separated_filepath)
        instrumental_track = data['track'].to(DEVICE) # GPU로 다시 이동
        sr = data['sr']

        # 2. 모노 변환 및 스펙트로그램 생성
        mono_instrumental_track = torch.mean(instrumental_track, dim=0, keepdim=True)
        long_spectrogram = mel_spectrogram(mono_instrumental_track, sample_rate=sr)
        if long_spectrogram is None:
            raise HTTPException(status_code=400, detail="Spectrogram conversion failed.")

        long_spectrogram = long_spectrogram.to(DEVICE)
        
        # 3. 슬라이딩 윈도우로 쿼리 벡터 생성
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

        with torch.no_grad():
            all_embeddings = MODEL(batch)
        query_vector = torch.mean(all_embeddings, dim=0, keepdim=True)

        # 4. Faiss 검색 수행
        results = SEARCH_ENGINE.search(query_vector.cpu().numpy(), top_k=5)

        return {
            "status": "success",
            "results": results
        }

    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 작업 완료 후 임시 파일 삭제
        if os.path.exists(separated_filepath):
            os.remove(separated_filepath)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)