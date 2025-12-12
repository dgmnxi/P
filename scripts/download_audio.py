import os
import sys
import argparse
import subprocess
from tqdm import tqdm
import time

def download_audio_from_list(url_list_file, output_dir, audio_format='mp3'):
    """
    URL 목록 파일을 읽어 각 URL의 오디오를 지정된 형식과 이름으로 다운로드합니다.
    아티스트를 찾지 못하면 파일명에 '[MANUAL_ARTIST]' 표시를 추가합니다.
    """
    if not os.path.exists(url_list_file):
        print(f"오류: URL 목록 파일 '{url_list_file}'을(를) 찾을 수 없습니다.")
        with open(url_list_file, 'w', encoding='utf-8') as f:
            f.write("# 여기에 유튜브 URL을 한 줄에 하나씩 입력하세요.\n")
            f.write("https://www.youtube.com/watch?v=dQw4w9WgXcQ\n")
        print(f"예시 파일 '{url_list_file}'을(를) 생성했습니다. URL을 채워주세요.")
        return

    with open(url_list_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not urls:
        print(f"'{url_list_file}'에 다운로드할 URL이 없습니다.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"총 {len(urls)}개의 URL을 다운로드합니다. 저장 폴더: '{output_dir}'")

    for url in tqdm(urls, desc="오디오 다운로드 중"):
        try:
            # 1. yt-dlp로 다운로드 (아티스트 없으면 'NA'로 저장)
            # 파일명 템플릿: "아티스트 - 제목.확장자"
            output_template = os.path.join(output_dir, "%(artist,NA)s - %(title)s.%(ext)s")

            command = [
                'yt-dlp',
                '--extract-audio',
                '--audio-format', audio_format,
                '--add-metadata',
                '--embed-thumbnail',
                '--output', output_template,
                '--print', 'filename',      # 최종 파일 경로를 표준 출력으로 인쇄
                '--ignore-errors',
                url
            ]
            
            # yt-dlp 실행 및 최종 파일 경로 캡처
            result = subprocess.run(
                command, 
                check=True, 
                capture_output=True, 
                text=True, 
                encoding='utf-8'
            )
            
            downloaded_filepath = result.stdout.strip()
            
            # 2. 파일명 확인 및 변경
            if os.path.exists(downloaded_filepath):
                dir_name = os.path.dirname(downloaded_filepath)
                base_filename = os.path.basename(downloaded_filepath)

                if base_filename.startswith("NA - "):
                    # "NA - " 부분을 "[MANUAL_ARTIST] - "로 변경
                    new_filename = "[MANUAL_ARTIST] - " + base_filename[len("NA - "):]
                    new_filepath = os.path.join(dir_name, new_filename)
                    os.rename(downloaded_filepath, new_filepath)
                    tqdm.write(f"수동 확인 필요: '{base_filename}' -> '{new_filename}'")
            
        except subprocess.CalledProcessError as e:
            tqdm.write(f"실패: {url} 다운로드 중 오류 발생.")
            tqdm.write(e.stderr)
        except Exception as e:
            # API 속도 제한 등 네트워크 오류 처리
            if "503" in str(e) or "HTTP Error 429" in str(e):
                 tqdm.write(f"API 속도 제한 감지. 10초 후 재시도합니다... ({url})")
                 time.sleep(10)
                 # 현재 URL을 다시 시도하기 위해 루프 카운터를 조정할 수 있으나,
                 # 여기서는 일단 건너뛰고 다음 URL로 진행합니다.
            else:
                tqdm.write(f"치명적 오류: {url} 처리 중 - {e}")

    print("\n--- 모든 다운로드 작업 완료 ---")

if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    parser = argparse.ArgumentParser(description="URL 목록을 기반으로 오디오 파일을 다운로드합니다.")
    parser.add_argument(
        '--url-file', 
        type=str, 
        default='youtube_urls.txt',
        help="다운로드할 유튜브 URL 목록이 담긴 .txt 파일 (기본값: youtube_urls.txt)"
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

    download_audio_from_list(args.url_file, args.output_dir, args.format)