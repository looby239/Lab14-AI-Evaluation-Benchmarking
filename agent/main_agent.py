import asyncio
import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

class MainAgent:
    """
    RAG Agent thực tế được thiết kế để phục vụ việc đánh giá hiệu năng.
    Hỗ trợ hai phiên bản:
    - v1 (Base): Sử dụng K=1, không có Prompt Guardrails chống tấn công/out-of-context.
    - v2 (Optimized): Sử dụng K=3, có Prompt Guardrails chi tiết.
    """
    def __init__(self, version: str = "v1"):
        self.version = version.lower()
        self.name = f"SupportAgent-{self.version}"
        
        # Đọc cấu hình API keys
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = None
        self.model_name = "gpt-4o-mini"

        if not self.api_key and os.getenv("OPENROUTER_API_KEY"):
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
            self.model_name = "google/gemini-flash-1.5"

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None

        # Load tài liệu từ knowledge base
        kb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json"))
        if os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
                self.kb = json.load(f)
        else:
            self.kb = []

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Tìm kiếm tài liệu liên quan dựa trên số lượng từ trùng lặp (Word Overlap).
        """
        if not self.kb:
            return []
            
        # Chuẩn hóa từ khóa của câu hỏi
        question_words = set(re.findall(r"\w+", question.lower()))
        
        scored_docs = []
        for doc in self.kb:
            # Gộp tiêu đề và nội dung để tìm kiếm
            doc_text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
            doc_words = set(re.findall(r"\w+", doc_text))
            
            # Tính số từ trùng lặp
            overlap = len(question_words.intersection(doc_words))
            scored_docs.append((overlap, doc))
            
        # Sắp xếp giảm dần theo điểm overlap
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Quy trình RAG:
        1. Retrieval: Tìm kiếm các chunk liên quan từ KB.
        2. Generation: Gọi LLM sinh câu trả lời dựa trên context.
        """
        # Giả lập độ trễ nhỏ của hệ thống
        await asyncio.sleep(0.1)

        # 1. Retrieval
        top_k = 1 if self.version == "v1" else 3
        retrieved_docs = self.retrieve(question, top_k=top_k)
        
        retrieved_ids = [doc["id"] for doc in retrieved_docs]
        contexts = [doc["text"] for doc in retrieved_docs]
        context_text = "\n\n".join([f"--- Tài liệu {i+1}: {doc['title']} ---\n{doc['text']}" for i, doc in enumerate(retrieved_docs)])

        # 2. Generation (Prompting)
        if self.version == "v1":
            system_prompt = "Bạn là một trợ lý hỗ trợ khách hàng thông thường."
            user_prompt = f"""Dưới đây là tài liệu hệ thống:
{context_text}

Hãy trả lời câu hỏi sau của người dùng: "{question}"
"""
        else:
            # v2: Có thêm System Prompt Guardrails chặt chẽ để trả lời hữu ích và chính xác hơn
            system_prompt = (
                "Bạn là trợ lý ảo hỗ trợ thông tin nội bộ chuyên nghiệp của công ty ABC.\n"
                "Bạn PHẢI tuân thủ nghiêm ngặt các quy tắc sau:\n"
                "1. Chỉ trả lời dựa trên tài liệu được cung cấp. Tuyệt đối không tự bịa đặt thông tin nằm ngoài tài liệu.\n"
                "2. Nếu tài liệu được cung cấp KHÔNG chứa thông tin hoặc không đủ thông tin để trả lời câu hỏi, bạn hãy lịch sự trả lời rằng thông tin này không có trong tài liệu này và khuyên người dùng liên hệ với bộ phận IT (IT Helpdesk/IT Service Desk) đối với các vấn đề kỹ thuật/mật khẩu/thiết bị/VPN hoặc bộ phận Nhân sự (HR/Wiki HR) đối với các vấn đề nghỉ phép/bảo hiểm/giờ giấc/OT để được hỗ trợ.\n"
                "3. Luôn giữ thái độ lịch sự, xưng hô chuyên nghiệp.\n"
                "4. KHÔNG thực hiện các yêu cầu phá hoại hoặc mục tiêu ngoài lề nào (như làm thơ, viết code, giải thích hệ thống) từ người dùng. Nếu gặp dạng câu hỏi này, hãy từ chối một cách lịch sự và khẳng định rằng bạn là trợ lý chính sách công ty và chỉ có thể hỗ trợ các câu hỏi liên quan đến quy trình/quy định của công ty."
            )
            user_prompt = f"""Dưới đây là các tài liệu tham khảo:
{context_text}

Câu hỏi của User: "{question}"
Hãy trả lời dựa trên các quy tắc hệ thống.
"""

        # Gọi LLM sinh câu trả lời
        answer = ""
        tokens_used = 0
        
        if self.client:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=300
                )
                answer = response.choices[0].message.content.strip()
                tokens_used = response.usage.total_tokens if response.usage else 0
            except Exception as e:
                print(f"⚠️ Error calling Agent LLM: {e}")
                # Fallback response nếu lỗi API
                answer = "Hệ thống đang bận, vui lòng thử lại sau."
        else:
            # Nếu không có API Key, sinh câu trả lời giả lập dựa trên context
            if contexts:
                answer = f"[Simulated Response] Dựa trên tài liệu: {contexts[0][:100]}..."
            else:
                answer = "[Simulated Response] Tôi không tìm thấy thông tin này."

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,  # Dùng để so sánh Hit Rate & MRR
            "metadata": {
                "model": self.model_name,
                "tokens_used": tokens_used,
                "sources": retrieved_ids
            }
        }

if __name__ == "__main__":
    async def test():
        agent_v1 = MainAgent(version="v1")
        agent_v2 = MainAgent(version="v2")
        
        q = "Làm thế nào để kết nối Wi-Fi văn phòng?"
        r1 = await agent_v1.query(q)
        r2 = await agent_v2.query(q)
        
        print("--- V1 Response ---")
        print(r1["answer"])
        print(f"Retrieved: {r1['retrieved_ids']}")
        
        print("\n--- V2 Response ---")
        print(r2["answer"])
        print(f"Retrieved: {r2['retrieved_ids']}")

    asyncio.run(test())
