import argparse
import os
import yt_dlp
import logging

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search_and_download_audio(query, limit, output_dir, cookies_file=None):
    """
    yt-dlp를 사용하여 유튜브에서 쿼리로 검색하고, 오디오를 다운로드합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"'{output_dir}' 디렉토리를 생성했습니다.")

    # --- 올바른 FFmpeg 경로 설정 ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 실제 압축 해제된 폴더 이름을 포함하여 정확한 경로를 지정합니다.
    ffmpeg_path = os.path.join(project_root, 'tools', 'ffmpeg', 'ffmpeg-master-latest-win64-gpl', 'bin')
    
    if os.path.exists(ffmpeg_path):
        logging.info(f"FFmpeg 경로를 설정합니다: {ffmpeg_path}")
    else:
        logging.warning(f"정확한 FFmpeg 경로를 찾을 수 없습니다: {ffmpeg_path}. 시스템 PATH에 의존합니다.")
        ffmpeg_path = 'ffmpeg' # 경로를 못찾으면 시스템에 설치된 ffmpeg를 사용하도록 시도

    # 검색과 다운로드를 위한 yt-dlp 옵션 설정
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'ignoreerrors': True,
        'quiet': True,
        'extract_flat': 'discard_in_playlist',
        'noplaylist': True,
        'ffmpeg_location': ffmpeg_path, # FFmpeg 경로 지정
    }

    # 쿠키 파일 경로가 제공되면 옵션에 추가
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts['cookiefile'] = cookies_file
        logging.info(f"'{cookies_file}'의 쿠키를 사용하여 다운로드합니다.")
    elif cookies_file:
        logging.warning(f"쿠키 파일을 찾을 수 없습니다: '{cookies_file}'. 쿠키 없이 진행합니다.")

    # 검색 쿼리 생성
    search_query = f"ytsearch{limit}:{query}"
    logging.info(f"'{search_query}'로 검색 및 다운로드를 시작합니다.")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(search_query, download=True)
            
            downloaded_count = 0
            if 'entries' in result:
                downloaded_count = len(result['entries'])
            
            logging.info("--- 다운로드 완료 ---")
            logging.info(f"총 {downloaded_count}개의 오디오를 성공적으로 다운로드 및 처리했습니다.")

    except Exception as e:
        logging.error(f"처리 중 오류 발생: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="유튜브에서 오디오를 검색하고 다운로드합니다.")
    parser.add_argument('--query', type=str, default="Billboard Hot 100", help="유튜브 검색어")
    parser.add_argument('--limit', type=int, default=10, help="다운로드할 최대 동영상 개수")
    parser.add_argument('--output-dir', type=str, default="data/raw", help="오디오 파일을 저장할 디렉토리")
    parser.add_argument('--cookies', type=str, default=None, help="유튜브 로그인 쿠키가 담긴 cookies.txt 파일 경로")
    args = parser.parse_args()

    search_and_download_audio(args.query, args.limit, args.output_dir, args.cookies)

if __name__ == '__main__':
    main()