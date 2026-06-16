import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge

async def run_benchmark_with_results(agent_version: str):
    print(f"[START] Khoi dong Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("[ERROR] Thieu data/golden_set.jsonl. Hay chay 'python data/synthetic_gen.py' truoc.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("[ERROR] File data/golden_set.jsonl rong. Hay tao it nhat 1 test case.")
        return None, None

    # Xác định phiên bản Agent thực tế cần chạy
    agent_sub_version = "v1" if "v1" in agent_version.lower() else "v2"
    agent = MainAgent(version=agent_sub_version)
    
    # Khởi tạo các engine đánh giá thực tế
    evaluator = RetrievalEvaluator()
    judge = LLMJudge()

    runner = BenchmarkRunner(agent, evaluator, judge)
    
    # Chạy song song bất đồng bộ với giới hạn batch_size=5 để tránh rate limit
    results = await runner.run_all(dataset, batch_size=5)

    total = len(results)
    if total == 0:
        return None, None

    # Tính toán các điểm số trung bình
    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    avg_hit_rate = sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total
    avg_agreement_rate = sum(r["judge"]["agreement_rate"] for r in results) / total

    summary = {
        "metadata": {
            "version": agent_version, 
            "total": total, 
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "metrics": {
            "avg_score": round(avg_score, 2),
            "hit_rate": round(avg_hit_rate, 2),
            "agreement_rate": round(avg_agreement_rate, 2)
        }
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")
    
    # Chạy benchmark cho phiên bản V2 đã tối ưu
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("[ERROR] Khong the chay Benchmark. Kiem tra lai data/golden_set.jsonl.")
        return

    print("\n--- KET QUA SO SANH (REGRESSION) ---")
    v1_score = v1_summary["metrics"]["avg_score"]
    v2_score = v2_summary["metrics"]["avg_score"]
    delta = v2_score - v1_score
    print(f"V1 Score: {v1_score:.2f}")
    print(f"V2 Score: {v2_score:.2f}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    # Cổng phê duyệt chất lượng (Regression Gate)
    # Đồng ý cập nhật nếu điểm V2 cao hơn V1, ngược lại từ chối release
    if delta > 0:
        print("[DECISION] CHAP NHAN BAN CAP NHAT (APPROVE)")
    else:
        print("[DECISION] TU CHOI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
