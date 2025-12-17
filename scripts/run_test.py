'''
실제 main.py의 함수 및 코드를 백엔드 서버와 통신없이 직접 테스트해보는 스크립트입니다.


'''



import argparse
import os
import torch
import sys
import time
import uuid
import yt_dlp
import torchaudio

# --- 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.model import resnet18_transfer_learning
from app.separator import separate_audio, mel_spectrogram
from app.search import VectorSearchEngine

# --- 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_WIDTH = 431
TEMP_DOWNLOAD_DIR = "temp_downloads_test"
COOKIE_FILE = "cookies.txt"

def main():
    parser = argparse.ArgumentParser(description="유튜브 링크로 유사도 검색을 테스트합니다.")
    parser.add_argument("youtube_url", type=str, help="검색할 유튜브 영상 링크")
    parser.add_argument("start_sec", type=float, help="검색할 구간의 시작 시간 (초)")
    parser.add_argument("end_sec", type=float, help="검색할 구간의 종료 시간 (초)")
    parser.add_argument("--instrument", type=str, default="other", help="분리할 악기 이름 (bass, drums, other, vocals)")
    parser.add_argument("--top_k", type=int, default=5, help="검색할 상위 결과 개수")
    args = parser.parse_args()

    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
    
    # --- 1. 리소스 로드 ---
    print("--- 리소스 로딩 시작 ---")
    model = resnet18_transfer_learning().to(DEVICE)
    model_path = os.path.join(project_root, "models/best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"모델 로드 완료: {model_path}")

    index_path = os.path.join(project_root, "indexes/timbre.index")
    metadata_path = os.path.join(project_root, "indexes/metadata.json")
    search_engine = VectorSearchEngine(index_path, metadata_path)
    print("검색 엔진 로드 완료")
    print("--- 리소스 로딩 완료 ---\n")

    # --- 2. 쿼리 오디오 처리 ---
    request_id = str(uuid.uuid4())
    download_path = os.path.join(TEMP_DOWNLOAD_DIR, f"{request_id}.%(ext)s")
    clipped_path = os.path.join(TEMP_DOWNLOAD_DIR, f"{request_id}_clipped.wav")
    actual_download_path = ""

    print(f"--- 쿼리 처리 시작: '{args.youtube_url}' ({args.start_sec:.2f}s ~ {args.end_sec:.2f}s) ---")
    try:
        # 2-1. 유튜브 오디오 다운로드
        print(f"Downloading audio from: {args.youtube_url}")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': download_path,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            'noplaylist': True,
        }
        if os.path.exists(COOKIE_FILE):
            print(f"Using cookie file: {COOKIE_FILE}")
            ydl_opts['cookiefile'] = COOKIE_FILE
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([args.youtube_url])
        
        actual_download_path = os.path.join(TEMP_DOWNLOAD_DIR, f"{request_id}.wav")

        # 2-2. 오디오 파일 자르기
        print(f"Clipping audio from {args.start_sec}s to {args.end_sec}s")
        waveform, sr = torchaudio.load(actual_download_path)
        start_frame, end_frame = int(args.start_sec * sr), int(args.end_sec * sr)
        clipped_waveform = waveform[:, start_frame:end_frame]
        torchaudio.save(clipped_path, clipped_waveform, sr)

        # 2-3. 음원 분리
        separation_start_time = time.time()
        stems, sr_sep = separate_audio(clipped_path, device=DEVICE)
        separation_end_time = time.time()
        print(f"음원 분리 소요 시간: {separation_end_time - separation_start_time:.2f}초")
        
        if stems is None or args.instrument not in stems:
            raise RuntimeError(f"Could not separate the '{args.instrument}' track.")
        
        instrumental_track = stems[args.instrument]

        # 2-4. 스펙트로그램 변환 및 쿼리 벡터 생성
        mono_track = torch.mean(instrumental_track, dim=0, keepdim=True)
        long_spectrogram = mel_spectrogram(mono_track, sample_rate=sr_sep).to(DEVICE)
        
        total_frames = long_spectrogram.shape[-1]
        window_size, stride = TARGET_WIDTH, TARGET_WIDTH // 2
        chunks = []
        if total_frames <= window_size:
            padded_chunk = torch.zeros(1, long_spectrogram.shape[1], window_size, device=DEVICE)
            padded_chunk[..., :total_frames] = long_spectrogram
            chunks.append(padded_chunk)
        else:
            for i in range(0, total_frames - window_size + 1, stride):
                chunks.append(long_spectrogram[..., i:i + window_size])
        
        if not chunks:
            raise RuntimeError("Could not extract any valid audio chunks.")

        batch = torch.cat(chunks, dim=0).unsqueeze(1)
        
        with torch.no_grad():
            all_embeddings = model(batch)
        query_vector = torch.mean(all_embeddings, dim=0, keepdim=True)
        print("쿼리 벡터 생성 완료")

    except Exception as e:
        print(f"쿼리 처리 중 오류 발생: {e}")
        return
    finally:
        # 임시 파일 정리
        if os.path.exists(actual_download_path): os.remove(actual_download_path)
        if os.path.exists(clipped_path): os.remove(clipped_path)


    # --- 3. 유사도 검색 및 결과 출력 ---
    print("\n--- 유사도 검색 결과 ---")
    search_start_time = time.time()
    results = search_engine.search(query_vector.cpu().numpy(), top_k=args.top_k)
    search_end_time = time.time()
    print(f"검색 소요 시간: {search_end_time - search_start_time:.4f}초")

    if not results:
        print("유사한 곡을 찾지 못했습니다.")
    else:
        for res in results:
            print(f"  - ID: {res['id']}, Similarity: {res['similarity']:.4f}")
            print(f"    Song: {res['song_name']}, Instrument: {res['instrument']}")
            print(f"    Time: {res['start_sec']:.2f}s ~ {res['end_sec']:.2f}s\n")

if __name__ == "__main__":
    main()