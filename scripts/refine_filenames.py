'''
download_audio.py 이후 실행시키는 스크립트입니다.

yt-dlp로 저장된 파일이 지정된 형식 (아티스트 - 제목.mp3) 으로 되어 있지 않은 경우,
이를 정제하는 역할을 합니다.

1. 아티스트 정리: 'A, B, C - Title.mp3' -> 'A - Title.mp3'
2. 제목 정리: 'Artist - Title (feat. X).mp3' -> 'Artist - Title.mp3'
   (단, 'remix', 'mix'가 포함된 괄호는 유지)

--dry-run 옵션을 주면 실제로 파일 이름을 변경하지 않고,
변경될 내역만 출력합니다.

** 실제 서비스에서는 사용되지 않음 **

'''


import os
import re
import argparse

def refine_filename(filename):
    """
    파일 이름을 정제하는 함수.
    1. 아티스트 정리: 'A, B, C - Title.mp3' -> 'A - Title.mp3'
    2. 제목 정리: 'Artist - Title (feat. X).mp3' -> 'Artist - Title.mp3'
       (단, 'remix', 'mix'가 포함된 괄호는 유지)
    """
    if ' - ' not in filename:
        return filename, None

    parts = filename.split(' - ')
    artist = parts[0]
    title_part = ' - '.join(parts[1:])

    # 1. 아티스트 정리
    refined_artist = artist.split(',')[0].strip()

    # 2. 제목 정리
    # 정규표현식을 사용하여 괄호 안의 내용을 찾음
    # 괄호 안에 'remix' 또는 'mix'가 없는 경우에만 제거
    def clean_parentheses(match):
        content = match.group(1)
        if 'remix' in content.lower() or 'mix' in content.lower():
            return f"({content})" # 유지
        return "" # 제거

    # .mp3 확장자를 잠시 분리
    if title_part.lower().endswith('.mp3'):
        title_base = title_part[:-4]
        extension = title_part[-4:]
    else:
        title_base = title_part
        extension = ''

    refined_title = re.sub(r'\s*\(([^)]*)\)', clean_parentheses, title_base).strip()
    
    # 여러 공백을 하나로
    refined_title = re.sub(r'\s+', ' ', refined_title)

    new_filename = f"{refined_artist} - {refined_title}{extension}"
    
    # 원본과 다른 경우에만 변경 사항을 반환
    if new_filename != filename:
        return new_filename, filename
    return new_filename, None

def main():
    parser = argparse.ArgumentParser(description="MP3 파일 이름을 정제합니다.")
    parser.add_argument("directory", type=str, help="MP3 파일이 있는 디렉토리 경로")
    parser.add_argument("--dry-run", action="store_true", help="실제로 파일 이름을 변경하지 않고 변경될 내역만 출력합니다.")
    args = parser.parse_args()

    directory = args.directory
    dry_run = args.dry_run

    if not os.path.isdir(directory):
        print(f"오류: 디렉토리를 찾을 수 없습니다: {directory}")
        return

    print(f"'{directory}' 디렉토리의 파일들을 스캔합니다...")
    if dry_run:
        print("--- [Dry Run] 모드로 실행합니다. 파일 이름이 실제로 변경되지 않습니다. ---")

    changed_files = []

    for filename in os.listdir(directory):
        if filename.lower().endswith('.mp3'):
            new_filename, old_filename = refine_filename(filename)
            if old_filename:
                old_path = os.path.join(directory, old_filename)
                new_path = os.path.join(directory, new_filename)
                
                if not dry_run:
                    try:
                        os.rename(old_path, new_path)
                        changed_files.append((old_filename, new_filename))
                    except Exception as e:
                        print(f"오류: '{old_filename}' 이름 변경 실패 - {e}")
                else:
                    changed_files.append((old_filename, new_filename))

    if changed_files:
        print("\n--- 이름 변경 내역 ---")
        for old, new in changed_files:
            print(f'"{old}"\n  -> "{new}"')
        print(f"\n총 {len(changed_files)}개의 파일 이름이 변경되었습니다.")
    else:
        print("\n변경할 파일이 없습니다.")

if __name__ == "__main__":
    main()
