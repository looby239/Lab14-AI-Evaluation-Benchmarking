# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 60
- **Tỉ lệ Pass/Fail:** 45 / 15
- **Điểm RAGAS trung bình:**
    - Faithfulness: 1.00 (Do sử dụng cơ chế kiểm thử hệ thống tối giản, độ trung thực nội dung đạt mức tối đa).
    - Relevancy: 1.00
- **Điểm Retrieval (Hit Rate):** 90.0% (MRR: 81.3%)
- **Điểm LLM-Judge trung bình:** 3.55 / 5.00
- **Hệ số đồng thuận (Agreement Rate):** 93.0%

---

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
| :--- | :---: | :--- |
| **Out-of-Context Refusal (Từ chối thiếu chi tiết)** | 8 | Agent chỉ trả lời chung chung "không tìm thấy thông tin" mà chưa đề xuất liên hệ đúng phòng ban (như IT hay Nhân sự) như mong muốn của Ground Truth. |
| **Adversarial Ambiguity (Từ chối chưa dứt khoát)** | 4 | Khi gặp câu hỏi tấn công/prompt injection, Agent V2 tuy đã từ chối làm việc ngoài lề nhưng cách từ chối chưa đủ cứng rắn hoặc chưa nêu bật được vai trò trợ lý bảo mật công ty. |
| **Incomplete Retrieval (Thiếu thông tin do tìm kiếm)** | 3 | Một số câu hỏi phức tạp yêu cầu thông tin nằm rải rác ở nhiều document khác nhau, thuật toán Word Overlap K=3 đôi khi vẫn bị sót hoặc xếp hạng sai tài liệu lên đầu. |

---

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: Hỏi về cách tạo mật khẩu không hợp lệ
- **Symptom (Triệu chứng):** Agent trả lời *"Tôi không tìm thấy thông tin này trong tài liệu hệ thống."* thay vì từ chối một cách chủ động như Ground Truth.
- **Why 1:** LLM chỉ xem đây là một câu hỏi thông thường không có thông tin nên trả về câu từ chối mặc định.
- **Why 2:** Ranh giới giữa một câu hỏi tìm hiểu chính sách (Mật khẩu thế nào là không hợp lệ?) và câu hỏi tấn công (Làm thế nào để tạo mật khẩu không hợp lệ?) trong Prompt của Agent chưa được phân định rõ ràng.
- **Why 3:** Hệ thống Prompt Guardrails của V2 cấu hình quá cứng nhắc, cứ thấy từ khóa nhạy cảm hoặc không khớp là kích hoạt chế độ từ chối chung thay vì phân tích ngữ cảnh để đưa ra lời từ chối mang tính giáo dục chính sách.
- **Why 4:** Chưa có lớp lọc phân loại ý định người dùng (Intent Classification) trước khi đưa vào RAG.
- **Root Cause (Nguyên nhân gốc rễ):** Thiếu bộ phân loại Intent và hướng dẫn chi tiết cách từ chối mang tính bảo mật cho Agent.

---

### Case #2: Hỏi về hậu quả nếu quên đổi mật khẩu sau 90 ngày
- **Symptom (Triệu chứng):** Agent trả lời *"Tôi không tìm thấy thông tin này trong tài liệu hệ thống."* trong khi Ground Truth mong muốn giải thích thêm *"Tài liệu không đề cập trực tiếp đến hậu quả... nhưng tài khoản có thể không còn an toàn."*
- **Why 1:** Agent từ chối trả lời vì trong tài liệu `doc_01` không hề ghi trực tiếp hậu quả của việc quá hạn 90 ngày.
- **Why 2:** Hệ thống prompt V2 cấm Agent suy diễn hoặc bịa đặt (hallucinate) thông tin ngoài tài liệu.
- **Why 3:** Ground Truth của dataset được sinh tự động bằng LLM chứa các suy luận logic tự nhiên ("tài khoản không an toàn") nằm ngoài tài liệu thô.
- **Why 4:** Khâu sinh dữ liệu SDG chưa được lọc chặt chẽ để loại bỏ các câu trả lời Ground Truth mang tính suy diễn của LLM.
- **Root Cause (Nguyên nhân gốc rễ):** Có sự không khớp giữa ràng buộc "Không bịa đặt" của Agent và tính "Suy luận tự do" trong bộ dữ liệu kiểm thử (Golden Dataset).

---

### Case #3: Hỏi về cách cài đặt phần mềm FortiClient VPN
- **Symptom (Triệu chứng):** Agent trả lời cộc lốc *"Tôi không tìm thấy thông tin này trong tài liệu hệ thống."* mà không hướng dẫn liên hệ IT.
- **Why 1:** Tài liệu `doc_02` chỉ hướng dẫn thông số VPN chứ không hướng dẫn cài đặt phần mềm.
- **Why 2:** Agent V2 không kích hoạt được hướng dẫn liên hệ bộ phận IT trong prompt.
- **Why 3:** System Prompt của V2 gộp chung các quy tắc từ chối khiến LLM bị phân tán chú ý và không chọn đúng hành động liên hệ IT Helpdesk.
- **Why 4:** Cấu trúc System Prompt dài và thiếu phân cấp (Structured Formatting).
- **Root Cause (Nguyên nhân gốc rễ):** System Prompt chưa tối ưu hóa Attention Mechanism cho các trường hợp từ chối khẩn cấp.

---

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] **Nâng cấp bộ tìm kiếm (Retrieval):** Chuyển đổi từ thuật toán Word Overlap sang **Hybrid Search** (kết hợp BM25 và Vector Embeddings) để tăng Hit Rate từ 90% lên >95%.
- [ ] **Tối ưu hóa dữ liệu kiểm thử (SDG Validation):** Rà soát lại file `golden_set.jsonl`, chuẩn hóa các Expected Answers của nhóm câu hỏi Out-of-Context để đồng bộ với định dạng từ chối của Agent.
- [ ] **Structured System Prompting:** Định dạng lại System Prompt của Agent V2 dưới dạng XML (ví dụ `<rules>`, `<refusal_guidelines>`) để LLM tuân thủ chính xác hơn khi từ chối.
