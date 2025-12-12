import os
import sys
import argparse
import subprocess
import time

def download_playlist(playlist_url, output_dir, audio_format='mp3', start_index=None, end_index=None):
    """
    주어진 유튜브 재생목록의 모든 오디오를 지정된 형식과 이름으로 다운로드합니다.
    - 특정 구간만 다운로드하는 기능을 지원합니다.
    - 디버그, 타임아웃, 아카이브 옵션으로 안정성을 높입니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    range_info = f" (구간: {start_index}-{end_index})" if start_index and end_index else ""
    print(f"재생목록 다운로드를 시작합니다: {playlist_url}{range_info}")
    print(f"저장 폴더: '{output_dir}'")

    try:
        output_template = os.path.join(output_dir, "%(artist,NA)s - %(title)s.%(ext)s")
        archive_file = os.path.join(output_dir, 'archive.txt')
        print(f"아카이브 파일 위치: '{archive_file}'")

        command = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', audio_format,
            '--add-metadata',
            '--embed-thumbnail',
            '--output', output_template,
            '--ignore-errors',
            '--verbose',
            '--socket-timeout', '30',
            '--download-archive', archive_file,
        ]

        # --- 구간 다운로드 옵션 추가 ---
        if start_index and end_index:
            command.extend(['--playlist-items', f'{start_index}-{end_index}'])

        script_dir = os.path.dirname(os.path.abspath(__file__))
        cookie_file = os.path.join(script_dir, 'cookies.txt')
        
        if os.path.exists(cookie_file):
            print(f"'{cookie_file}' 파일을 쿠키로 사용합니다.")
            command.extend(['--cookies', cookie_file])
        else:
            print("쿠키 파일(scripts/cookies.txt)을 찾을 수 없습니다. 없이 진행합니다.")
        
        command.append(playlist_url)

        print("\n--- yt-dlp 실행 (디버그 모드) ---")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        if process.returncode != 0:
            print(f"\n경고: yt-dlp 실행이 비정상적으로 종료되었습니다 (종료 코드: {process.returncode}).")

    except Exception as e:
        print(f"\n치명적 오류 발생: {e}")

    print("\n--- 모든 작업 완료 ---")

if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    parser = argparse.ArgumentParser(description="유튜브 재생목록의 모든 오디오를 다운로드합니다.")
    parser.add_argument('playlist_url', type=str, help="다운로드할 유튜브 재생목록의 전체 URL")
    parser.add_argument('--output-dir', type=str, default=os.path.join(PROJECT_ROOT, "data/raw"), help="다운로드한 오디오 파일을 저장할 디렉토리")
    parser.add_argument('--format', type=str, default='mp3', help="오디오 포맷 (mp3, wav, flac 등)")
    # --- 구간 지정을 위한 인자 추가 ---
    parser.add_argument('--start', type=int, help="다운로드를 시작할 재생목록 인덱스 (예: 1)")
    parser.add_argument('--end', type=int, help="다운로드를 종료할 재생목록 인덱스 (예: 100)")
    
    args = parser.parse_args()

    download_playlist(args.playlist_url, args.output_dir, args.format, args.start, args.end)