'''
유사한 음원 쌍을 찾는 스크립트입니다.
- FAISS 인덱스에서 벡터를 검색하여 유사도가 높은 쌍을 출력합니다.
- 메타데이터 파일에서 각 벡터에 대한 정보를 조회합니다.

** 테스트 파일임. 실 서비스에서는 사용되지 않음
'''


import faiss
import json
import numpy as np
import os
import sys
import random

def main():
    # --- 경로 설정 ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    index_path = os.path.join(project_root, "indexes/timbre.index")
    metadata_path = os.path.join(project_root, "indexes/metadata.json")

    # --- 설정 ---
    SIMILARITY_THRESHOLD = 0.95  # 찾고자 하는 유사도 임계값 (0.0 ~ 1.0)
    MAX_PAIRS_TO_FIND = 10 # 찾을 최대 쌍의 개수
    
    # --- 리소스 로드 ---
    print("--- 리소스 로딩 시작 ---")
    if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
        print(f"오류: 인덱스 또는 메타데이터 파일을 찾을 수 없습니다.")
        return

    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"인덱스 로드 완료 (총 {index.ntotal}개 벡터)")
        print(f"메타데이터 로드 완료 (총 {len(metadata)}개 항목)")
        print(f"유사도 {SIMILARITY_THRESHOLD:.2f} 이상인 쌍을 찾습니다.")
        print("--- 리소스 로딩 완료 ---\n")
    except Exception as e:
        print(f"리소스 로딩 중 오류 발생: {e}")
        return

    found_pairs = 0
    
    # 모든 벡터를 순회하면 너무 오래 걸리므로, 인덱스에서 랜덤하게 샘플을 뽑아 검사합니다.
    # 중복 검사를 피하기 위해 조회한 인덱스를 기록합니다.
    total_vectors = index.ntotal
    if total_vectors < 2:
        print("인덱스에 벡터가 충분하지 않습니다.")
        return
        
    # 랜덤하게 시작점을 섞어서 매번 다른 결과를 얻도록 합니다.
    indices_to_check = list(range(total_vectors))
    random.shuffle(indices_to_check)

    for i in indices_to_check:
        if found_pairs >= MAX_PAIRS_TO_FIND:
            print(f"최대 {MAX_PAIRS_TO_FIND}개의 쌍을 모두 찾았으므로 탐색을 종료합니다.")
            break
            
        # i번째 벡터를 쿼리로 사용하여, 가장 가까운 이웃 2개를 찾습니다.
        # k=2인 이유는 첫 번째 결과는 항상 자기 자신(유사도 1)이기 때문입니다.
        query_vector = np.array([index.reconstruct(i)])
        # IndexFlatIP에서는 search 결과가 '유사도'가 됩니다.
        similarities, neighbors = index.search(query_vector, k=2)
        
        # 두 번째 이웃 (가장 가까운 '다른' 벡터)
        neighbor_id = neighbors[0][1]
        similarity = similarities[0][1]
        
        # 유사도가 임계값 이상인지 확인
        if similarity >= SIMILARITY_THRESHOLD:
            
            found_pairs += 1
            
            # --- 최종 수정: 인덱스를 문자열로 변환하여 조회 ---
            str_i = str(i)
            str_neighbor_id = str(neighbor_id)

            if str_i not in metadata or str_neighbor_id not in metadata:
                print(f"[경고] 인덱스 {str_i} 또는 {str_neighbor_id}에 해당하는 메타데이터를 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # 메타데이터 조회
            meta1 = metadata[str_i]
            meta2 = metadata[str_neighbor_id]
            # --- 수정 끝 ---
            
            print(f"--- 쌍 #{found_pairs} 발견! (유사도: {similarity:.4f}) ---")
            print(f"  [A] Song: {meta1['song_name']}, Inst: {meta1['instrument']}, Time: {meta1['start_sec']:.2f}s ~ {meta1['end_sec']:.2f}s")
            print(f"  [B] Song: {meta2['song_name']}, Inst: {meta2['instrument']}, Time: {meta2['start_sec']:.2f}s ~ {meta2['end_sec']:.2f}s\n")

    if found_pairs == 0:
        print(f"유사도 {SIMILARITY_THRESHOLD:.2f} 이상의 쌍을 찾지 못했습니다.")

if __name__ == "__main__":
    main()