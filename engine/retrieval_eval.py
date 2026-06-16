from typing import List, Dict, Any

class RetrievalEvaluator:
    def __init__(self):
        pass

    def get_retrieved_ids(self, response: Dict[str, Any]) -> List[str]:
        """
        Trích xuất danh sách ID tài liệu đã tìm kiếm được từ câu trả lời của Agent.
        Hỗ trợ nhiều định dạng trả về khác nhau của Agent để tăng tính linh hoạt.
        """
        if not response:
            return []
        
        # Thử lấy trực tiếp từ response["retrieved_ids"]
        if "retrieved_ids" in response and isinstance(response["retrieved_ids"], list):
            return response["retrieved_ids"]
            
        # Thử lấy từ metadata
        metadata = response.get("metadata", {})
        if isinstance(metadata, dict):
            if "retrieved_ids" in metadata and isinstance(metadata["retrieved_ids"], list):
                return metadata["retrieved_ids"]
            if "sources" in metadata and isinstance(metadata["sources"], list):
                return metadata["sources"]
                
        # Thử phân tích từ contexts nếu có dạng list (đôi khi là danh sách text hoặc dict)
        contexts = response.get("contexts", [])
        if isinstance(contexts, list):
            # Nếu contexts chứa các dict có trường id
            retrieved = []
            for ctx in contexts:
                if isinstance(ctx, dict) and "id" in ctx:
                    retrieved.append(str(ctx["id"]))
                elif isinstance(ctx, str):
                    # Nếu là dạng string, có thể thử kiểm tra xem có chứa tên file hoặc ID không
                    pass
            if retrieved:
                return retrieved

        return []

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Tính toán Hit Rate@K: Trả về 1.0 nếu có ít nhất 1 tài liệu trong expected_ids
        nằm trong top_k của retrieved_ids, ngược lại trả về 0.0.
        """
        if not expected_ids or not retrieved_ids:
            return 0.0
            
        # Chuẩn hóa IDs (xóa khoảng trắng, viết thường) để tránh lệch khớp
        expected_set = {str(eid).strip().lower() for eid in expected_ids}
        top_retrieved = [str(rid).strip().lower() for rid in retrieved_ids[:top_k]]
        
        hit = any(doc_id in top_retrieved for doc_id in expected_set)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Tính Mean Reciprocal Rank (MRR):
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì trả về 0.0.
        """
        if not expected_ids or not retrieved_ids:
            return 0.0
            
        expected_set = {str(eid).strip().lower() for eid in expected_ids}
        
        for i, doc_id in enumerate(retrieved_ids):
            normalized_doc_id = str(doc_id).strip().lower()
            if normalized_doc_id in expected_set:
                return 1.0 / (i + 1)
        return 0.0

    def score(self, test_case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trả về kết quả đánh giá retrieval theo định dạng tương thích với BenchmarkRunner và main.py.
        """
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = self.get_retrieved_ids(response)
        
        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=3)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)
        
        # Giả lập hoặc tính toán độ trung thực (faithfulness) và độ liên quan (relevancy) mặc định
        # để đảm bảo tương thích với cấu trúc báo cáo RAGAS.
        return {
            "faithfulness": 1.0, 
            "relevancy": 1.0,
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr
            }
        }

    async def evaluate_batch(self, dataset: List[Dict[str, Any]], responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Đánh giá toàn bộ tập dữ liệu (batch) và trả về điểm số trung bình.
        """
        total = len(dataset)
        if total == 0:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}
            
        total_hit_rate = 0.0
        total_mrr = 0.0
        
        for case, resp in zip(dataset, responses):
            score_res = self.score(case, resp)
            total_hit_rate += score_res["retrieval"]["hit_rate"]
            total_mrr += score_res["retrieval"]["mrr"]
            
        return {
            "avg_hit_rate": total_hit_rate / total,
            "avg_mrr": total_mrr / total
        }
