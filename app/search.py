'''
Faiss를 이용한 벡터 검색 엔진 구현

timbre.index를 통한 검색을 수행

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
            
    def search(self, query_vector: np.ndarray, top_k: int = 5, exclude_title: str = None, instruments: list[str] = None) -> list:
        """
        주어진 쿼리 벡터와 가장 유사한 top_k개의 결과를 반환합니다.
        - 최종 결과에 동일한 곡(title)이 중복되지 않도록 합니다.
        - 만약 한 곡 내에서 연속되는 구간이 여러 개 발견되면, 이들을 합쳐서 하나의 결과로 만듭니다.
        - instruments가 지정된 경우, 해당 악기만 필터링합니다.
        """


        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        
        faiss.normalize_L2(query_vector)

        # 중복 제거 및 병합을 위해 충분히 많은 후보군을 검색합니다.
        search_k = top_k * 20
        
        try:
            similarities, ids = self.index.search(query_vector, search_k)
        except Exception as e:
            print(f"Faiss search error: {e}")
            return []
        


        # 1. 모든 후보군을 title 기준으로 그룹화
        candidates_by_title = {}
        for i in range(len(ids[0])):
            result_id = str(ids[0][i])
            if result_id not in self.metadata:
                continue

            meta = self.metadata[result_id]
            title = meta.get("title")
            instrument = meta.get("instrument")

            # 쿼리 곡 자체는 제외
            if not title or title == exclude_title:
                continue
            
            # 악기 필터링
            if instruments and instrument not in instruments:
                continue

            if title not in candidates_by_title:
                candidates_by_title[title] = []
            
            candidates_by_title[title].append({
                "id": result_id,
                "similarity": float(similarities[0][i]),
                "artist": meta.get("artist"),
                "title": title,
                "instrument": meta.get("instrument"),
                "start_sec": meta.get("start_sec"),
                "end_sec": meta.get("end_sec"),
            })

        # 2. 각 곡별로 구간 병합 수행 및 대표 결과 생성
        merged_results = []
        for title, segments in candidates_by_title.items():
            # 유사도 순으로 정렬하여 가장 유사한 구간을 기준으로 삼음
            segments.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 가장 유사도가 높은 구간을 대표로 설정
            best_segment = segments[0].copy()
            
            # 시간 순으로 재정렬하여 병합 준비
            segments.sort(key=lambda x: x['start_sec'])

            # 병합 로직
            merged_segment = None
            for seg in segments:
                if merged_segment is None:
                    merged_segment = seg.copy()
                    continue
                
                # 구간이 겹치거나 바로 연속될 경우 (1초 허용)
                if seg['start_sec'] <= merged_segment['end_sec'] + 1.0:
                    merged_segment['end_sec'] = max(merged_segment['end_sec'], seg['end_sec'])
                else:
                    # 연속되지 않으면 병합 중단 (가장 유사한 구간 주변만 병합)
                    break
            
            # 병합된 시간 정보와 가장 높았던 유사도를 최종 결과로 사용
            final_segment = best_segment
            final_segment['start_sec'] = merged_segment['start_sec']
            final_segment['end_sec'] = merged_segment['end_sec']
            
            merged_results.append(final_segment)

        # 3. 최종 결과를 유사도 순으로 정렬하여 top_k개 반환
        merged_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return merged_results[:top_k]