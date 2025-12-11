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
        특정 song_name을 결과에서 제외할 수 있습니다.
        """
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        
        faiss.normalize_L2(query_vector)

        # 필터링을 위해 충분한 수의 결과를 가져옵니다.
        search_k = top_k * 5 if exclude_song_name else top_k
        
        similarities, ids = self.index.search(query_vector, search_k)
        
        results = []
        for i in range(len(ids[0])):
            if len(results) >= top_k:
                break

            result_id = str(ids[0][i])
            similarity = float(similarities[0][i])
            
            if result_id in self.metadata:
                meta = self.metadata[result_id]
                
                # 제외할 song_name이 있고, 현재 결과의 song_name과 일치하면 건너뜁니다.
                if exclude_song_name and meta.get("song_name") == exclude_song_name:
                    continue

                results.append({
                    "id": result_id,
                    "similarity": similarity,
                    "song_name": meta.get("song_name"),
                    "instrument": meta.get("instrument"),
                    "start_sec": meta.get("start_sec"),
                    "end_sec": meta.get("end_sec"),
                })
        return results