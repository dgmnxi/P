'''
모델 훈련에 필요한 데이터 셋을 준비하는 스크립트 입니다.
원본 오디오 파일을 로드하고, Demucs를 사용하여 악기별로 분리한 후,
각 악기별로 지정된 길이로 분할하고 Mel 스펙토그램으로 변환하여 저장합니다.
이 때, 각 악기별로 5초 단위로 분할하여 저장하며 , 저장 형식은 PyTorch 텐서(.pt / 1X128x413) 입니다.

'''


import argparse
import os
import torch
import torchaudio
from glob import glob
import logging
from tqdm import tqdm
import sys
import shutil


# 절대 경로 참조
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# 프로젝트 루트를 sys.path에 추가하여 app 모듈을 임포트
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from app.separator import separate_audio, mel_spectrogram
except ImportError as e:
    print(f"Error: {e}")
    print("Could not import from 'app.separator'. Make sure you are running this script from the project root directory.")
    sys.exit(1)

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 설정 ---
# 5초, 44.1kHz, hop_length=512 기준으로 계산된 최종 스펙트로그램 너비
TARGET_WIDTH = 431
RMS_THRESHOLD = 1e-4  # 침묵으로 간주할 RMS 에너지 임계값 (이 값보다 작으면 저장 안 함)


def process_and_save(audio_path, output_dir, segment_duration=5.0, device='cuda'):
    """
    하나의 오디오 파일을 처리하여 분리, 분할, Mel Spectrogram 변환 후 저장합니다.
    모든 결과물의 크기를 동일하게 보장하고, 침묵 구간은 저장하지 않습니다.
    """
    try:
        separated_stems, sr = separate_audio(audio_path, device=device)
        original_filename = os.path.splitext(os.path.basename(audio_path))[0]

        for instrument, tensor in separated_stems.items():
            num_samples_per_segment = int(segment_duration * sr)
            
            # separate_audio가 스테레오를 반환하므로, 여기서 모노로 변환합니다.
            if tensor.dim() > 1 and tensor.shape[0] == 2:
                tensor = tensor.mean(dim=0) 

            segments = list(tensor.split(num_samples_per_segment, dim=-1))

            # 마지막 세그먼트가 너무 짧으면 버리기
            if segments and segments[-1].shape[-1] < num_samples_per_segment:
                segments.pop()

            if not segments:
                continue

            instrument_dir = os.path.join(output_dir, instrument)
            os.makedirs(instrument_dir, exist_ok=True)

            for i, seg in enumerate(segments):
                # 길이가 5초보다 짧으면 무시 (안전장치)
                if seg.shape[-1] < num_samples_per_segment:
                    continue
                
                # --- 침묵 구간 제거 로직 ---
                # RMS 에너지를 계산합니다.
                rms = torch.sqrt(torch.mean(seg.pow(2)))
                
                # RMS 값이 임계값보다 낮으면 침묵으로 간주하고 건너뜁니다.
                if rms < RMS_THRESHOLD:
                    continue
                # --- 로직 끝 ---

                # unsqueeze(0)로 채널 차원 추가
                mel_spec = mel_spectrogram(seg.unsqueeze(0), sample_rate=sr)

                # 스펙트로그램의 너비가 TARGET_WIDTH와 다르면 보정합니다.
                if mel_spec.shape[2] > TARGET_WIDTH:
                    mel_spec = mel_spec[:, :, :TARGET_WIDTH]
                elif mel_spec.shape[2] < TARGET_WIDTH:
                    tqdm.write(f"경고: '{original_filename}'의 {i}번째 조각이 목표 크기({TARGET_WIDTH})보다 작아 건너뜁니다. (크기: {mel_spec.shape[2]})")
                    continue
                
                save_filename = f"{original_filename}_{i:03d}.pt"
                save_path = os.path.join(instrument_dir, save_filename)
                torch.save(mel_spec, save_path)
        return True, None
    except Exception as e:
        logging.error(f"'{os.path.basename(audio_path)}' 처리 중 오류 발생: {e}", exc_info=True)
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="오디오 파일을 전처리하여 학습용 데이터셋을 생성합니다.")
    # --- 경로 문제 해결을 위한 수정: 기본 경로를 절대 경로로 설정 ---
    parser.add_argument('--input-dir', type=str, default=os.path.join(PROJECT_ROOT, "data/raw"), help="처리할 원본 오디오 파일이 있는 디렉토리")
    parser.add_argument('--output-dir', type=str, default=os.path.join(PROJECT_ROOT, "data/processed"), help="처리된 Mel Spectrogram 텐서를 저장할 디렉토리")
    parser.add_argument('--segment-duration', type=float, default=5.0, help="오디오를 분할할 시간 (초 단위)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="PyTorch 장치 (cuda 또는 cpu)")
    args = parser.parse_args()

    logging.info(f"입력 디렉토리: {args.input_dir}")
    logging.info(f"출력 디렉토리: {args.output_dir}")

    audio_files = []
    supported_exts = ["**/*.mp3", "**/*.wav", "**/*.flac", "**/*.m4a", "**/*.ogg"]
    for ext in supported_exts:
        # recursive=True를 사용하여 하위 폴더까지 모두 검색
        pattern = os.path.join(args.input_dir, ext)
        audio_files.extend(glob(pattern, recursive=True))

    # 중복 제거
    audio_files = sorted(list(set(audio_files)))

    if not audio_files:
        logging.warning(f"'{args.input_dir}'에서 오디오 파일을 찾을 수 없습니다. 파일이 있는지, 확장자가 올바른지 확인해주세요.")
        return

    logging.info(f"총 {len(audio_files)}개의 오디오 파일을 처리합니다.")
    
    # --- 중복 처리 확인 > 중복 시 skip ---
    processed_files_map = {}
    for instrument in ["bass", "drums", "other", "vocals"]:
        instrument_dir = os.path.join(args.output_dir, instrument)
        if os.path.exists(instrument_dir):
            # 예: "song1_001.pt" -> "song1"
            processed_files_map[instrument] = { "_".join(f.split('_')[:-1]) for f in os.listdir(instrument_dir) }

    success_count = 0
    fail_count = 0
    # tqdm을 사용하여 파일 처리 진행 상황을 표시합니다.
    with tqdm(total=len(audio_files), desc="오디오 파일 처리 중") as pbar:
        for file_path in audio_files:
            original_filename = os.path.splitext(os.path.basename(file_path))[0]
            
            # --- 추가: 모든 악기에 대해 이미 처리되었는지 확인 ---
            is_already_processed = True
            # bass, drums, other 악기가 모두 처리되었는지 확인
            for instrument in ["bass", "drums", "other"]: 
                if instrument not in processed_files_map or original_filename not in processed_files_map[instrument]:
                    is_already_processed = False
                    break
            
            if is_already_processed:
                pbar.update(1)
                tqdm.write(f"'{original_filename}'은(는) 이미 처리되었습니다. 건너뜁니다.")
                continue

            success, error_msg = process_and_save(file_path, args.output_dir, args.segment_duration, args.device)
            if not success:
                tqdm.write(f"오류: '{os.path.basename(file_path)}' 처리 실패 - {error_msg}")
            pbar.update(1)

    logging.info("--- 전처리 완료 ---")
    logging.info(f"성공: {success_count}개 파일, 실패: {fail_count}개 파일")

if __name__ == '__main__':
    main()