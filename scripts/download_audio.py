import os
import sys
import argparse
import subprocess
from tqdm import tqdm
import time

def download_playlist(playlist_url, output_dir, audio_format='mp3'):
    """
    주어진 유튜브 재생목록의 모든 오디오를 지정된 형식과 이름으로 다운로드합니다.
    - 'scripts/cookies.txt' 파일이 있으면 자동으로 사용합니다.
    - 아티스트를 찾지 못하면 파일명에 '[MANUAL_ARTIST]' 표시를 추가합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"재생목록 다운로드를 시작합니다: {playlist_url}")
    print(f"저장 폴더: '{output_dir}'")

    try:
        # 파일명 템플릿: "아티스트 - 제목.확장자"
        # yt-dlp가 아티스트를 찾지 못하면 'NA' (Not Available)을 사용합니다.
        output_template = os.path.join(output_dir, "%(artist,NA)s - %(title)s.%(ext)s")

        command = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', audio_format,
            '--add-metadata',
            '--embed-thumbnail',
            '--output', output_template,
            '--print', 'filename',      # 최종 파일 경로를 표준 출력으로 인쇄
            '--ignore-errors',      # 오류 발생 시 계속 진행
        ]

        # --- cookies.txt 파일 확인 및 추가 ---
        # 스크립트가 있는 디렉토리 기준으로 cookies.txt를 찾음
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cookie_file = os.path.join(script_dir, 'cookies.txt')
        
        if os.path.exists(cookie_file):
            print(f"'{cookie_file}' 파일을 쿠키로 사용합니다.")
            command.extend(['--cookies', cookie_file])
        else:
            print("쿠키 파일(scripts/cookies.txt)을 찾을 수 없습니다. 없이 진행합니다.")
        
        # 마지막에 URL 추가
        command.append(playlist_url)

        # yt-dlp 실행 및 최종 파일 경로 캡처
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True, 
            encoding='utf-8'
        )
        
        # 출력된 모든 파일 경로에 대해 처리
        downloaded_files = result.stdout.strip().splitlines()
        
        print(f"\n다운로드 완료. 총 {len(downloaded_files)}개의 파일명 후처리 작업을 시작합니다.")
        for filepath in tqdm(downloaded_files, desc="파일명 후처리 중"):
            if os.path.exists(filepath):
                dir_name = os.path.dirname(filepath)
                base_filename = os.path.basename(filepath)

                if base_filename.startswith("NA - "):
                    new_filename = "[MANUAL_ARTIST] - " + base_filename[len("NA - "):]
                    new_filepath = os.path.join(dir_name, new_filename)
                    try:
                        os.rename(filepath, new_filepath)
                        tqdm.write(f"수동 확인 필요: '{base_filename}' -> '{new_filename}'")
                    except OSError as e:
                        tqdm.write(f"파일명 변경 실패: {e}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n오류: yt-dlp 실행 중 오류가 발생했습니다.")
        print(e.stderr)
    except Exception as e:
        print(f"\n치명적 오류 발생: {e}")

    print("\n--- 모든 작업 완료 ---")

if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    parser = argparse.ArgumentParser(description="유튜브 재생목록의 모든 오디오를 다운로드합니다.")
    parser.add_argument(
        'playlist_url', 
        type=str, 
        help="다운로드할 유튜브 재생목록의 전체 URL"
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=os.path.join(PROJECT_ROOT, "data/raw_new"),
        help="다운로드한 오디오 파일을 저장할 디렉토리"
    )
    parser.add_argument(
        '--format', 
        type=str, 
        default='mp3',
        help="오디오 포맷 (mp3, wav, flac 등)"
    )
    
    args = parser.parse_args()

    download_playlist(args.playlist_url, args.output_dir, args.format)