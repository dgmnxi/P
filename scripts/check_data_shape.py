'''
데이터 폴더 내의 .pt 파일들의 shape을 검사하는 스크립트입니다.
모든 파일이 동일한 shape을 가지는지 확인하여 데이터 준비 과정에서의 문제를 조기에 발견할 수 있습니다.

실행용이 아닌 테스트 파일
'''


import os
import torch
from glob import glob
from tqdm import tqdm
import collections

# --- 경로 설정 ---
# 이 스크립트 파일의 위치를 기준으로 프로젝트 루트의 절대 경로를 계산합니다.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")

def check_shapes():
    """
    data/processed 폴더의 모든 .pt 파일의 shape을 확인하고,
    크기가 다른 파일이 있는지 검사합니다.
    """
    print(f"'{DATA_DIR}' 폴더의 데이터를 검사합니다...")
    
    all_files = glob(os.path.join(DATA_DIR, '**', '*.pt'), recursive=True)

    if not all_files:
        print("검사할 .pt 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    shape_counts = collections.defaultdict(list)
    
    # tqdm을 사용하여 진행 상황 표시
    for file_path in tqdm(all_files, desc="파일 검사 중"):
        try:
            tensor = torch.load(file_path, map_location='cpu')
            shape_counts[tensor.shape].append(file_path)
        except Exception as e:
            print(f"\n'{file_path}' 파일 로드 중 오류 발생: {e}")

    print("\n--- 검사 결과 ---")
    if len(shape_counts) == 1:
        shape = list(shape_counts.keys())[0]
        print(f"✅ 모든 {len(all_files)}개 파일의 크기가 {shape} (으)로 동일합니다. 데이터에 문제가 없습니다.")
    else:
        print(f"❌ 경고: {len(shape_counts)}개의 다른 크기가 발견되었습니다!")
        print("각 크기별 파일 개수:")
        for shape, files in shape_counts.items():
            print(f"  - 크기 {shape}: {len(files)}개")
        
        print("\n크기가 다른 파일 예시:")
        for shape, files in shape_counts.items():
            print(f"  - 크기 {shape} 파일: {files[0]}")

if __name__ == '__main__':
    check_shapes()
