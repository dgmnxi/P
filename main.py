# main.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import torch
import shutil
import os
import numpy as np

# 우리가 만든 모듈들 임포트
from app.model import resnet18_transfer_learning
from app.separator import extract_and_transform_frame
from app.search import VectorSearchEngine

app = FastAPI(title="Music Timbre Search AI Server")

# --- 전역 변수 및 설정 ---
MODEL = None
SEARCH_ENGINE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 학습 시 사용했던 입력 크기 (5초 분량의 Mel Spectrogram 프레임 수)
# 이 값은 app/separator.py의 mel_spectrogram 설정에 따라 달라질 수 있습니다.
# 44100 * 5 / 512 = 430.6 -> 431 프레임
TRAIN_INPUT_FRAMES = 431 

@app.on_event("startup")
async def load_resources():
    global MODEL, SEARCH_ENGINE
    
    # 1. 모델 로드
    print("Loading AI Model...")
    MODEL = resnet18_transfer_learning(output_dim=128, freeze_features=False).to(DEVICE)
    
    model_path = "models/audio_resnet_best.pth"
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Warning: Model weights not found at {model_path}. Using initial transfer learning weights.")
        
    MODEL.eval()
    
    # 2. 검색 엔진 로드
    print("Loading Search Engine...")
    index_path = "indexes/timbre.index"
    metadata_path = "indexes/metadata.json"
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        SEARCH_ENGINE = VectorSearchEngine(index_path, metadata_path)
    else:
        print("Warning: Index or metadata file not found. Search will fail.")

'''
API 호출 부 
입력 : 오디오 파일, 악기 이름, 시작 시간, 종료 시간
출력 : 유사한 음악 리스트 top_k = 5/ 각 항목은 다음과 같은 딕셔너리 형태
            "status": "success",
            "results": 
                        {
                            "id": result_id : int
                            "distance": distance : float,
                            "song_name": song_name : str,
                            "instrument": instrument : str,
                            "start_sec": start_sec: float,
                            "end_sec": end_sec: float
                        }
'''
@app.post("/api/search")
async def search_music(
    file: UploadFile = File(...), 
    start_sec: float = Form(...), 
    end_sec: float = Form(...),
    instrument: str = Form(...)
):
    """
    오디오 파일과 구간 정보를 받아 유사한 음악을 검색합니다.
    (슬라이딩 윈도우 적용)
    """
    temp_filename = f"temp_{file.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. 사용자가 요청한 전체 구간의 스펙토그램을 추출
        long_spectrogram = extract_and_transform_frame(temp_filename, instrument, start_sec, end_sec, device=DEVICE)
        
        if long_spectrogram is None:
            raise HTTPException(status_code=400, detail="Audio processing failed")

        long_spectrogram = long_spectrogram.to(DEVICE)
        
        # --- 슬라이딩 윈도우 로직 ---
        total_frames = long_spectrogram.shape[-1]
        window_size = TRAIN_INPUT_FRAMES
        stride = window_size // 2  # 윈도우를 50%씩 겹치게 이동

        # 2. 윈도우를 움직여 여러 개의 조각(chunk)을 배치로 만듦
        chunks = []
        if total_frames <= window_size:
            # 입력이 훈련 크기보다 작거나 같으면, 그대로 사용
            chunks.append(long_spectrogram)
        else:
            # 입력이 더 길면, 슬라이딩하며 조각 추출
            for i in range(0, total_frames - window_size + 1, stride):
                chunk = long_spectrogram[..., i:i + window_size]
                chunks.append(chunk)
        
        if not chunks:
             raise HTTPException(status_code=400, detail="Could not extract any valid audio chunks.")

        batch = torch.cat(chunks, dim=0)

        # 3. 모델로 모든 조각을 한 번에 추론
        with torch.no_grad():
            all_embeddings = MODEL(batch)

        # 4. 추론된 모든 임베딩 벡터를 평균내어 최종 쿼리 벡터 생성
        query_vector = torch.mean(all_embeddings, dim=0, keepdim=True)
        
        # --------------------------

        # 5. Faiss 검색 수행
        if SEARCH_ENGINE:
            results = SEARCH_ENGINE.search(query_vector.cpu().numpy(), top_k=5)
        else:
            results = []

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