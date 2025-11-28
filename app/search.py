'''
Faiss를 이용한 벡터 검색 엔진 구현

'''

# app/search.py
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
            
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list:
        """
        주어진 쿼리 벡터와 가장 유사한 top_k개의 결과를 반환합니다.
        """
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        
        # Faiss 검색 수행
        distances, ids = self.index.search(query_vector, top_k)
        
        results = []
        for i in range(top_k):
            result_id = str(ids[0][i])
            distance = float(distances[0][i])
            
            if result_id in self.metadata:
                meta = self.metadata[result_id]
                results.append({
                    "id": result_id,
                    "distance": distance,
                    "song_name": meta.get("song_name"),
                    "instrument": meta.get("instrument"),
                    "start_sec": meta.get("start_sec"),
                    "end_sec": meta.get("end_sec"),
                })
        return results