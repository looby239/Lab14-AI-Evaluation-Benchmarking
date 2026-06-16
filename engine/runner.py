import asyncio
import time
from typing import List, Dict, Any

class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chạy một ca kiểm thử đơn lẻ:
        1. Gọi Agent để sinh câu trả lời.
        2. Chạy Retrieval Evaluator để đánh giá Hit Rate và MRR.
        3. Chạy Multi-Judge để đánh giá chất lượng câu trả lời.
        """
        start_time = time.perf_counter()
        
        # 1. Gọi Agent
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time
        
        # 2. Chạy Retrieval/RAGAS metrics
        eval_scores = self.evaluator.score(test_case, response)
        
        # 3. Chạy Multi-Judge
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"], 
            response["answer"], 
            test_case["expected_answer"]
        )
        
        # Tính toán trạng thái pass/fail (Đạt nếu điểm trung bình Judge từ 3.0 trở lên)
        status = "pass" if judge_result["final_score"] >= 3.0 else "fail"
        
        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": round(latency, 3),
            "ragas": eval_scores,
            "judge": judge_result,
            "status": status
        }

    async def run_all(self, dataset: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Chạy song song toàn bộ các test case bằng asyncio.gather kết hợp với asyncio.Semaphore
        để kiểm soát lưu lượng concurrency (Rate Limit).
        """
        semaphore = asyncio.Semaphore(batch_size)
        
        async def sem_run(case: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.run_single_test(case)
                except Exception as e:
                    # Trả về kết quả mặc định nếu có sự cố để tránh sập toàn bộ luồng chạy
                    print(f"❌ Error running test case '{case.get('question')}': {e}")
                    return {
                        "test_case": case.get("question", ""),
                        "agent_response": "Error: API call failed.",
                        "latency": 0.0,
                        "ragas": {
                            "faithfulness": 0.0,
                            "relevancy": 0.0,
                            "retrieval": {"hit_rate": 0.0, "mrr": 0.0}
                        },
                        "judge": {
                            "final_score": 1.0,
                            "agreement_rate": 1.0,
                            "reasoning": f"Execution error: {str(e)}"
                        },
                        "status": "fail"
                    }

        tasks = [sem_run(case) for case in dataset]
        return await asyncio.gather(*tasks)
