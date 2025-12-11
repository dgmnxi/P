'''
Faiss를 이용한 벡터 검색 엔진 구현

'''
import faiss
import json
import numpy as np

class VectorSearchEngine:
    def __init__(self, index_path: str, metadata_path: str):
        """
        Faiss 인덱스와 메타데이터를 로드합니다.
        """
        print(f"Loading Faiss index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
    def search(self, query_vector: np.ndarray, top_k: int = 5, exclude_song_name: str = None) -> list:
        """
        주어진 쿼리 벡터와 가장 유사한 top_k개의 결과를 반환합니다.
        - 특정 song_name을 결과에서 제외할 수 있습니다.
        - 동일 곡의 연속된 구간은 합쳐서 표현합니다.
        """
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        
        faiss.normalize_L2(query_vector)

        # 병합 및 필터링을 위해 충분히 많은 후보군을 검색합니다.
        search_k = top_k * 10 
        
        similarities, ids = self.index.search(query_vector, search_k)
        
        # 1. 초기 후보군 생성 및 필터링
        candidates = []
        for i in range(len(ids[0])):
            result_id = str(ids[0][i])
            
            if result_id not in self.metadata:
                continue

            meta = self.metadata[result_id]
            
            # 제외할 곡 필터링
            if exclude_song_name and meta.get("song_name") == exclude_song_name:
                continue
            
            candidates.append({
                "id": result_id,
                "similarity": float(similarities[0][i]),
                "song_name": meta.get("song_name"),
                "instrument": meta.get("instrument"),
                "start_sec": meta.get("start_sec"),
                "end_sec": meta.get("end_sec"),
            })

        # 2. 곡별로 그룹화 및 구간 병합
        merged_results = {}
        for cand in candidates:
            key = (cand["song_name"], cand["instrument"])
            if key not in merged_results:
                merged_results[key] = []
            
            merged_results[key].append(cand)

        final_results = []
        for key, group in merged_results.items():
            # 시작 시간 기준으로 정렬
            group.sort(key=lambda x: x["start_sec"])
            
            if not group:
                continue

            # 첫 번째 구간으로 병합 시작
            current_merge = group[0].copy()

            for i in range(1, len(group)):
                next_item = group[i]
                # 구간이 겹치거나 바로 연속될 경우 (약간의 오차 허용)
                if next_item["start_sec"] <= current_merge["end_sec"] + 1.0:
                    # 종료 시간 확장
                    current_merge["end_sec"] = max(current_merge["end_sec"], next_item["end_sec"])
                    # 유사도는 더 높은 값으로 갱신
                    current_merge["similarity"] = max(current_merge["similarity"], next_item["similarity"])
                else:
                    # 병합된 구간을 결과에 추가하고 새 병합 시작
                    final_results.append(current_merge)
                    current_merge = next_item.copy()
            
            # 마지막으로 병합된 구간 추가
            final_results.append(current_merge)

        # 3. 최종 결과 정렬 및 top_k 반환
        # 유사도가 높은 순으로 정렬
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return final_results[:top_k]