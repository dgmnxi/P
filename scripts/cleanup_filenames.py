import os
import sys
import argparse
import musicbrainzngs
import glob
import re
from tqdm import tqdm

# --- MusicBrainz API 설정 ---
musicbrainzngs.set_useragent(
    "P-Project-File-Cleaner",
    "0.1",
    "https://github.com/dgmnxi/P"
)

def sanitize_filename(name):
    """파일 이름에 사용할 수 없는 문자를 제거하거나 대체합니다."""
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def cleanup_filenames(data_dir, apply_changes=False):
    """
    지정된 디렉토리의 오디오 파일 이름을 MusicBrainz API를 통해 정리합니다.
    """
    supported_exts = ["*.mp3", "*.wav", "*.flac", "*.m4a"]
    audio_files = []
    for ext in supported_exts:
        audio_files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))

    if not audio_files:
        print(f"'{data_dir}'에서 정리할 오디오 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(audio_files)}개의 파일을 확인합니다.")
    
    rename_plan = []

    for file_path in tqdm(audio_files, desc="파일 정보 조회 중"):
        try:
            dir_name = os.path.dirname(file_path)
            original_filename = os.path.basename(file_path)
            filename_no_ext, extension = os.path.splitext(original_filename)

            query = filename_no_ext.replace('NA -', '').strip()
            query = re.sub(r'\s*\(.*?\)|\[.*?\]', '', query).strip()

            result = musicbrainzngs.search_recordings(query=query, limit=1)
            
            if not result.get('recording-list'):
                tqdm.write(f"결과 없음: '{original_filename}'에 대한 정보를 찾지 못했습니다.")
                continue

            top_result = result['recording-list'][0]
            
            # --- 아티스트 이름 추출 로직 수정 ---
            artist_name = "Unknown"
            artist_credit = top_result.get('artist-credit')

            # 1. artist-credit가 리스트인 경우 (가장 일반적)
            if isinstance(artist_credit, list):
                artist_name = " & ".join([cred.get('name', '') or cred.get('artist', {}).get('name', '') for cred in artist_credit])
            # 2. artist-credit가 문자열인 경우
            elif isinstance(artist_credit, str):
                artist_name = artist_credit
            
            if not artist_name or artist_name == "Unknown":
                 tqdm.write(f"아티스트 정보 없음: '{original_filename}'")
                 continue
            # --- 로직 수정 끝 ---

            title = top_result.get('title', filename_no_ext)

            sanitized_artist = sanitize_filename(artist_name)
            sanitized_title = sanitize_filename(title)
            new_filename = f"{sanitized_artist} - {sanitized_title}{extension}"
            
            if new_filename != original_filename:
                rename_plan.append({
                    "dir": dir_name,
                    "old": original_filename,
                    "new": new_filename
                })

        except Exception as e:
            tqdm.write(f"오류 발생: '{original_filename}' 처리 중 오류 - {e}")

    if not rename_plan:
        print("\n모든 파일명이 이미 올바릅니다. 변경할 파일이 없습니다.")
        return

    print(f"\n--- 파일명 변경 계획 ({len(rename_plan)}개) ---")
    for plan in rename_plan:
        print(f"'{plan['old']}'  ->  '{plan['new']}'")

    if apply_changes:
        print("\n--- 실제 파일명 변경을 시작합니다. ---")
        for plan in tqdm(rename_plan, desc="파일명 변경 중"):
            try:
                old_path = os.path.join(plan['dir'], plan['old'])
                new_path = os.path.join(plan['dir'], plan['new'])
                os.rename(old_path, new_path)
            except Exception as e:
                tqdm.write(f"오류: '{plan['old']}' 파일명 변경 실패 - {e}")
        print("--- 파일명 변경 완료 ---")
    else:
        print("\n'--apply' 플래그가 없으므로 미리보기만 실행했습니다.")
        print("실제로 파일명을 변경하려면 아래 명령어를 실행하세요:")
        print(f"python {os.path.basename(__file__)} --data-dir \"{data_dir}\" --apply")


if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    parser = argparse.ArgumentParser(description="MusicBrainz API를 사용하여 오디오 파일명을 정리합니다.")
    parser.add_argument('--data-dir', type=str, default=os.path.join(PROJECT_ROOT, "data/raw"), help="정리할 오디오 파일이 있는 디렉토리")
    parser.add_argument('--apply', action='store_true', help="이 플래그가 있으면 실제로 파일명을 변경합니다.")
    
    args = parser.parse_args()

    cleanup_filenames(args.data_dir, args.apply)