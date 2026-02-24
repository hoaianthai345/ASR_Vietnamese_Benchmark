---
draft: false
---
### **1. Làm sao biết hướng đi chính của bài là khả thi và có ý nghĩa đóng góp: 
#### "model selection thường bị dẫn dắt bởi offline WER" và bỏ qua latency/throughput/compute?**

(a) Trong thực hành nghiên cứu/leaderboard, **WER** là chỉ số trung tâm; **efficiency/latency** thường không được báo cáo nhất quán

- Bài [_Open ASR Leaderboard_](https://arxiv.org/html/2510.06961v3) nêu rõ rằng đánh giá ASR lâu nay chủ yếu tập trung vào độ chính xác (WER) và **các chỉ số hiệu năng/hiệu quả (như RTF) thường “hiếm khi được báo cáo”**, do đó họ đề xuất chuẩn hoá đánh giá kèm RTF/RTFx để so sánh “accuracy–efficiency”.
    

(b) Các hệ thống “production-oriented” và các nghiên cứu triển khai **phải** đo latency/RTF/throughput vì đây là ràng buộc vận hành thực tế

- [WeNet](https://www.isca-archive.org/interspeech_2021/yao21_interspeech.pdf) (toolkit định hướng production cho cả streaming và non-streaming) nhấn mạnh việc báo cáo **runtime benchmark** gồm **RTF và latency** như một phần thiết kế hướng triển khai.
    
- [Pratap et al. (Interspeech)](https://ronan.collobert.com/pub/2020_scalingcnnasr_interspeech.pdf) khi bàn về triển khai ASR ở quy mô lớn cũng định nghĩa ba ràng buộc: **throughput**, **RTF/latency**, và **WER** – tức là “accuracy” chỉ là một phần trong bài toán triển khai.
    
- Một ví dụ khác ([MDPI](https://www.mdpi.com/2079-9292/10/21/2697)) mô tả rõ **RTF là thước đo latency** khi đánh giá ASR trên thiết bị/edge và báo cáo phân phối RTF.
    

Từ (a) và (b), ta có thể lập luận chặt chẽ trong Introduction như sau:

> “Trong khi WER là thước đo phổ biến để xếp hạng mô hình, các hệ thống production/streaming yêu cầu thêm các chỉ số hiệu năng (RTF/latency/throughput), nhưng các chỉ số này thường không được báo cáo đồng nhất trong đánh giá mô hình.”


#### **Vì sao “khoảng trống này bị khuếch đại với tiếng Việt” (domain shift, accent variability, numeric/abbrev)?**

 (a) Domain shift: read speech vs spontaneous/noisy speech là khác phân phối mạnh

- VIVOS được mô tả là một corpus tiếng Việt (khoảng 15 giờ) phục vụ ASR và có tính chất **giọng đọc/chuẩn bị sẵn** (read/recorded speech).
    
- Common Voice (vi) (dù không trích trực tiếp ở kết quả trên) về bản chất là dữ liệu cộng đồng với **đa dạng người nói**, thường kéo theo accent/noise và điều kiện thu âm không kiểm soát; vì vậy khi đặt VIVOS (clean/read) cạnh Common Voice (noisy/diverse), đã có một “đòn bẩy” hợp lý để chứng minh domain shift trong benchmark.
    

(b) Accent variability: các nghiên cứu/corpus tiếng Việt thường nhấn mạnh đa dạng vùng miền/giọng

- Một corpus tiếng Việt quy mô lớn gần đây được giới thiệu và **phân tích theo phương ngữ (regional dialect)**, cho thấy bản thân cộng đồng nghiên cứu cũng xem “đa dạng vùng miền” là biến số quan trọng trong ASR tiếng Việt.
    
- VietMed (ASR tiếng Việt trong miền y tế) cũng nhấn mạnh việc bao phủ **nhiều accent** và đánh giá baseline theo bối cảnh chuyên biệt, củng cố luận điểm rằng tiếng Việt thường gặp vấn đề “accent + domain”.
    

(c) Numeric/abbreviated expressions: vì sao có thể đưa vào (nhưng phải viết cẩn trọng)

Phần “frequent numeric/abbreviated expressions” là đúng hướng trong triển khai (CS, tài chính, giáo dục), nhưng để **tránh over-claim** khi chưa có citation trực tiếp cho tiếng Việt, nên diễn đạt theo kiểu:

- “Các ứng dụng tiếng Việt trong thực tế thường chứa nhiều **số/ngày-tháng/đơn vị/viết tắt**, làm tăng độ phức tạp chuẩn hoá văn bản và đánh giá.”  
    Có thể dùng nguồn về **text normalization/evaluation standardization** để “neo” lập luận:
    
-  [_Open ASR Leaderboard_](https://arxiv.org/html/2510.06961v3) coi việc **chuẩn hoá text normalization** là điều kiện để so sánh công bằng (điều này liên quan trực tiếp đến số/viết tắt/chuẩn hoá).
    

Cách để tăng giá trị cho bài là **tự định nghĩa một error category** “numbers/units/abbreviations” trong taxonomy (C4) và báo cáo tần suất/lỗi trên sample 200–500 câu. Khi đó, **chính kết quả** trở thành bằng chứng (thay vì chỉ là claim).

### 2. Contribution của bài?

Trình bày một hệ thống benchmark định hướng triển khai cho bài toán nhận dạng tiếng nói tiếng Việt theo chế độ streaming, vượt ra ngoài cách đánh giá chỉ dựa trên độ chính xác ngoại tuyến. 
Cụ thể, 
- (i)  đề xuất một giao thức đánh giá streaming có khả năng tái lập cùng với hiện thực tham chiếu mở, chuẩn hóa các thành phần như chia khối (chunking), ngữ cảnh nhìn trước (look-ahead), xử lý chồng lắp và các chỉ số đo lường có xét đến độ trễ; 
- (ii)  báo cáo mối quan hệ đánh đổi giữa WER–độ trễ–RTF và các biên Pareto trên nhiều họ mô hình ASR tiêu biểu, từ đó hỗ trợ việc lựa chọn mô hình theo định hướng triển khai thực tế; 
- (iii)  xây dựng một bảng đánh giá độ bền (robustness scorecard) nhằm định lượng mức suy giảm độ chính xác dưới các ràng buộc streaming và những thay đổi phân phối dữ liệu mang tính thực tiễn; và 
- (iv)  thực hiện phân tích lỗi chuyên biệt cho tiếng Việt, qua đó hình thành một hệ phân loại lỗi cùng các insight có tính hành động phục vụ triển khai trong môi trường thực tế.

### 3. Pipeline nghiên cứu và tính khả thi

1. **Tổng quan pipeline nghiên cứu**

Nghiên cứu được thiết kế theo hướng **đánh giá hệ thống (system-oriented benchmark)** thay vì huấn luyện mô hình mới. Pipeline gồm bốn giai đoạn chính: chuẩn bị dữ liệu, mô phỏng streaming và suy luận, đánh giá đa tiêu chí, và phân tích kết quả. Cách tiếp cận này giúp giảm đáng kể chi phí tính toán, đồng thời vẫn đảm bảo giá trị học thuật và tính ứng dụng cao.

---

2. **Giai đoạn 1 – Chuẩn bị dữ liệu và mô hình**

Nhóm sử dụng các tập dữ liệu tiếng Việt công khai:
- [VIVOS](https://huggingface.co/datasets/AILAB-VNUHCM/vivos): clean/read baseline.
- [VLSP2020](https://huggingface.co/datasets/doof-ferb/vlsp2020_vinai_100h): benchmark chuẩn VN + có spontaneous.
- [Viet YouTube ASR v2](https://huggingface.co/datasets/linhtran92/viet_youtube_asr_corpus_v2): “in-the-wild” shift, đa dạng domain.
- [Speech-MASSIVE_vie](https://huggingface.co/datasets/doof-ferb/Speech-MASSIVE_vie): Intent/assistant speech → hợp streaming use-case (assistant-like).
Tổng: 6–10 giờ (subset).

Dữ liệu âm thanh được chuẩn hóa về tần số lấy mẫu, kênh mono và định dạng thống nhất. Song song đó, văn bản tham chiếu được chuẩn hóa theo cùng một quy tắc chấm điểm nhằm đảm bảo so sánh công bằng giữa các mô hình. 

Nhóm chỉ sử dụng các mô hình ASR có sẵn checkpoint công khai, không thực hiện huấn luyện lại:
- [**khanhld/chunkformer-ctc-large-vie**](https://huggingface.co/khanhld/chunkformer-ctc-large-vie) (110M)  
    Lý do: streaming-native/chunk-based, là “nhân vật chính” của câu chuyện WER–Latency. Đây gần như phải có để benchmark streaming.
- [**vinai/PhoWhisper-large**](https://huggingface.co/vinai/PhoWhisper-large) (1.55B)  
    Lý do: offline upper bound tiếng Việt rất mạnh; dùng làm “accuracy ceiling” và đối chứng với chi phí/RTF. Reviewer sẽ hỏi “đỉnh offline là gì?” — model này trả lời câu hỏi đó.
- [**nguyenvulebinh/wav2vec2-base-vietnamese-250h**](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h) (95M)  
    Lý do: baseline CTC phổ biến, nhẹ, dễ chạy streaming chunk simulation. Đây là baseline “sạch” để chứng minh chunkformer thật sự đẩy frontier.
- [**openai/whisper-large-v3**](https://huggingface.co/openai/whisper-large-v3) (1.55B)  
    Lý do: đối chứng “general multilingual SOTA” vs “Vietnamese-specialized” (PhoWhisper). Chỉ cần chạy **offline** là đủ, không nhất thiết streaming. Mục tiêu là chứng minh “fine-tuned Vietnamese matters”.
- [**faster-whisper medium** ](Systran/faster-whisper-medium) 
	Không phải model mới, nhưng là **deployment variant**.  
	Lý do: system track rất thích “cost-aware”: cùng một model nhưng inference engine/quantization khác nhau → khác RTF, khác latency. Bạn có thể tạo một mini-study “deployment knob” mà không cần training.

**Tính khả thi:** Giai đoạn này chủ yếu là xử lý dữ liệu và suy luận offline, phù hợp để thực hiện trên Google Colab với GPU A100 hoặc thậm chí CPU cho các mô hình nhỏ.

---

 3. **Giai đoạn 2 – Mô phỏng streaming và suy luận**

Nhóm xây dựng một cơ chế mô phỏng streaming bằng cách chia audio thành các đoạn ngắn (chunk) với kích thước khác nhau và thiết lập ngữ cảnh nhìn trước (look-ahead). Các cấu hình streaming này cho phép đánh giá mô hình dưới các mức độ trễ khác nhau, từ gần thời gian thực đến bán thời gian thực.

Quá trình suy luận được thực hiện theo hai chế độ: offline (toàn bộ ngữ cảnh) và streaming (giới hạn ngữ cảnh). Việc này giúp so sánh trực tiếp ảnh hưởng của ràng buộc streaming lên độ chính xác và hiệu năng.

**Tính khả thi:** Không cần huấn luyện hay fine-tune, chỉ chạy suy luận nhiều lần với cấu hình khác nhau. Mỗi cấu hình có thể chạy độc lập, dễ phân công cho các thành viên trong nhóm và phù hợp với thời lượng phiên Colab giới hạn.

---

4. **Giai đoạn 3 – Đánh giá và đo lường đa tiêu chí**

Kết quả suy luận được đánh giá bằng nhiều chỉ số, bao gồm Word Error Rate (WER), độ trễ suy luận ước tính, hệ số thời gian thực (Real-Time Factor – RTF), và mức suy giảm độ chính xác khi chuyển từ offline sang streaming. Đối với WER, nhóm áp dụng kỹ thuật bootstrap để ước lượng khoảng tin cậy, giúp tăng độ tin cậy thống kê của kết quả.

Việc đánh giá đa tiêu chí cho phép phân tích mối quan hệ đánh đổi giữa độ chính xác và hiệu năng, thay vì chỉ dựa trên một chỉ số đơn lẻ.

**Tính khả thi:** Các phép đo này có thể thực hiện hoàn toàn bằng script Python, không yêu cầu tài nguyên tính toán lớn và có thể chạy song song.

---

5. **Giai đoạn 4 – Phân tích và rút insight**

Từ các chỉ số đo được, nhóm xây dựng các biểu đồ thể hiện mối quan hệ WER–độ trễ và xác định các cấu hình Pareto tối ưu. Ngoài ra, nhóm tiến hành phân tích độ bền của mô hình dưới các điều kiện khác nhau như độ dài câu nói hoặc mức độ nhiễu.

Một tập mẫu lỗi được chọn ngẫu nhiên để phân tích thủ công, từ đó xây dựng hệ phân loại lỗi đặc thù cho tiếng Việt, phục vụ việc rút ra các khuyến nghị triển khai thực tế.

**Tính khả thi:** Phân tích này thiên về xử lý số liệu và đánh giá định tính, phù hợp với năng lực của sinh viên và không phụ thuộc nhiều vào phần cứng.

---

6. **Đánh giá tổng thể tính khả thi**

Toàn bộ pipeline không yêu cầu huấn luyện mô hình hay xử lý dữ liệu quy mô lớn. Các bước đều có thể thực hiện bằng công cụ mã nguồn mở và dữ liệu công khai. Nhờ thiết kế mô-đun, có thể phân chia công việc theo từng giai đoạn, thực hiện song song và kiểm soát tốt tiến độ. Do đó, nghiên cứu này **hoàn toàn khả thi**, ngay cả khi chỉ có quyền truy cập hạn chế vào tài nguyên tính toán.