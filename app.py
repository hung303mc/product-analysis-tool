import streamlit as st
import google.generativeai as genai
import csv
import time
import json
import os
from datetime import datetime
import io
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Product Analysis Tool",
    page_icon="🚀",
    layout="wide"
)

# --- GOOGLE API SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (AttributeError, KeyError):
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    st.error("API Key not found! Please set it in Streamlit's secrets or as an environment variable and restart.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# --- PROMPT TEMPLATE ---
DEFAULT_PROMPT = """
Mày là một nhà chiến lược tìm kiếm cơ hội trong thương mại điện tử. Mục tiêu chính của mày là xác định các sản phẩm "winner" MỚI cho một doanh nghiệp online linh hoạt.

BỐI CẢNH & QUY TẮC KINH DOANH:
1.  **Mô hình Logistics:** Vận chuyển hàng không (Air freight) từ Trung Quốc về kho ở Mỹ. Sản phẩm không lưu kho dài hạn.
2.  **GIỚI HẠN CÂN NẶNG NGHIÊM NGẶT:** Sản phẩm > 4 lbs (~1.8 kg) sẽ tự động bị 'Bỏ qua' (Skip). Lý tưởng là dưới 3 lbs (~1.4 kg).
3.  **Lợi nhuận là trên hết:** Sử dụng chi phí vận chuyển tham khảo (~$15 cho 1lb, ~$25 cho 2lbs, v.v.) để ước tính lợi nhuận ròng tiềm năng.
4.  **THÀNH CÔNG BAN ĐẦU (để tham khảo, không phải giới hạn):** Doanh nghiệp đã có thành công bước đầu với ngách "Các thiết bị cho thời tiết nóng" và "Phụ kiện thời trang mùa hè". Hãy dùng thông tin này làm ngữ cảnh để nhận diện các mô thức thành công (ví dụ: sản phẩm độc đáo giải quyết vấn đề), nhưng TUYỆT ĐỐI KHÔNG chấm điểm thấp cho các sản phẩm thuộc ngành hàng mới. Mục tiêu chính là tìm ra NGÁCH THÀNH CÔNG TIẾP THEO.
5.  **Mục tiêu cốt lõi:** Tìm kiếm các sản phẩm độc đáo, có tiềm năng viral, có thể tìm nguồn từ 1688.

NHIỆM VỤ:
Phân tích LÔ sản phẩm dưới đây với tư duy cởi mở. Mục tiêu chính của mày là xác định các sản phẩm "winner" tiềm năng tiếp theo, bất kể chúng thuộc danh mục nào. Cung cấp một "overall_summary" chiến lược và sau đó đánh giá TỪNG sản phẩm.

Dữ liệu lô sản phẩm (định dạng CSV):
{product_batch_csv}

Phân tích và trả lời trong một định dạng JSON hợp lệ duy nhất. KHÔNG thêm bất kỳ văn bản nào bên ngoài đối tượng JSON chính.

Cấu trúc phản hồi JSON phải như sau:
{{
  "overall_summary": "string (Tóm tắt chiến lược của mày về lô hàng, nhấn mạnh vào các CƠ HỘI MỚI tốt nhất. Cung cấp phần này bằng tiếng Việt.)",
  "product_analyses": [
    {{
      "asin": "string (ASIN của sản phẩm)",
      "classification": "string (Giữ nguyên các thuật ngữ tiếng Anh này: Winner, High Potential, Potential, Skip)",
      "reason": "string (Lý do toàn diện của mày, giải thích tại sao đây có thể là một sản phẩm win, ngay cả khi nó ở trong một ngách mới. Cung cấp phần này bằng tiếng Việt.)",
      "viral_potential": "string (Giữ nguyên các thuật ngữ tiếng Anh này: Low, Medium, High)",
      "uniqueness_score": "number (1-10, 1 là hàng phổ thông, 10 là cực kỳ độc đáo)",
      "market_trend_score": "number (1-10, sản phẩm này phù hợp với các xu hướng hiện tại hoặc mới nổi ở thị trường Mỹ như thế nào)",
      "logistics_fit": "string (Giữ nguyên các thuật ngữ tiếng Anh này: Good, Acceptable, Poor)",
      "estimated_shipping_cost": "number (Ước tính của mày bằng USD)",
      "profit_potential": "string (Giữ nguyên các thuật ngữ tiếng Anh này: Low, Medium, High, Very High)",
      "risks": ["string (Liệt kê các rủi ro chính bằng tiếng Việt.)"]
    }}
  ]
}}
"""

# --- CORE FUNCTIONS (Logic xử lý chính) ---

def analyze_product_batch_api(product_batch_list, model, prompt_template):
    if not product_batch_list:
        return None
    
    output = io.StringIO()
    fieldnames = product_batch_list[0].keys()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(product_batch_list)
    csv_string = output.getvalue()

    prompt = prompt_template.format(product_batch_csv=csv_string)
    
    try:
        response = model.generate_content(prompt)
        if response.parts:
            cleaned_text = response.text.strip().replace('```json', '').replace('```', '')
            return json.loads(cleaned_text)
        else:
            return None
    except Exception as e:
        st.error(f"Lỗi API hoặc JSON: {e}")
        return None

# --- STREAMLIT APP UI ---

st.title("🚀 Công Cụ Phân Tích Sản Phẩm")
st.info("Upload file CSV sản phẩm, chỉnh sửa prompt nếu cần, sau đó nhấn 'Bắt đầu Phân tích'.")

# <<< MỤC MỚI 1: HƯỚNG DẪN SỬ DỤNG >>>
with st.expander("📖 Xem hướng dẫn sử dụng (Workflow)", expanded=False):
    st.markdown("""
    **Công cụ này được thiết kế để làm việc với file CSV được xuất ra từ tool Black Box của Helium 10.**

    #### **Quy trình làm việc:**
    1.  **📥 Lấy File Input:**
        * Vào **Helium 10 Black Box**, lọc sản phẩm theo các tiêu chí mày muốn.
        * Nhấn nút **"Export Data"** để tải về file CSV.

    2.  **⬆️ Tải File Lên Đây:**
        * Kéo và thả hoặc nhấn vào nút 'Browse files' để tải file CSV vừa download từ Helium 10 lên.

    3.  **⚙️ (Tùy chọn) Tinh Chỉnh:**
        * Ở thanh bên trái, mày có thể chọn Model AI hoặc điều chỉnh 'Batch Size' (số sản phẩm xử lý mỗi lần).
        * Nếu cần, mày có thể sửa đổi Prompt để thay đổi cách AI phân tích.

    4.  **🚀 Bắt Đầu Phân Tích:**
        * Nhấn vào nút **'Bắt đầu Phân tích'** và chờ công cụ xử lý.
        * Công cụ sẽ đọc file của mày, gửi dữ liệu cho AI phân tích, và tạo ra một file kết quả mới.

    5.  **📄 Xem & Tải Kết Quả:**
        * File output sẽ bao gồm các cột dữ liệu gốc quan trọng và các cột phân tích mới từ AI (như `classification`, `reason`, `viral_potential`...).
        * Nhấn nút **'Tải về file kết quả'** để lưu file CSV đã được làm giàu thông tin về máy.
    """)


# -- Sidebar chứa các cài đặt --
st.sidebar.header("Cài đặt")
model_name = st.sidebar.selectbox(
    "Chọn Model Gemini", 
    ('gemini-2.5-pro', 'gemini-1.5-pro', 'gemini-1.5-flash'),
    index=0, 
    help="2.5 Pro cho chất lượng phân tích cao nhất."
)
batch_size = st.sidebar.slider(
    "Số sản phẩm mỗi lô (Batch Size)", 
    min_value=5, max_value=20, value=10,
    help="Số lượng sản phẩm gửi cho AI trong một lần. Lớn hơn giúp AI so sánh tốt hơn."
)

# -- Ô sửa prompt --
with st.expander("📝 Chỉnh sửa Prompt (Nâng cao)"):
    prompt_text = st.text_area("Prompt Template:", value=DEFAULT_PROMPT, height=400)

# -- Khu vực Upload file --
uploaded_file = st.file_uploader("Chọn file CSV của mày:", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Đã tải lên thành công file '{uploaded_file.name}' với {len(df)} sản phẩm.")
        
        st.dataframe(df.head())

        # <<< MỤC MỚI 2: ƯỚC TÍNH THỜI GIAN >>>
        num_products = len(df)
        # Dựa trên ước tính thực tế: 1 lô 10 sản phẩm với model Pro mất ~60 giây
        time_per_batch_seconds = 60 
        num_batches = (num_products + batch_size - 1) // batch_size
        total_seconds = num_batches * time_per_batch_seconds
        
        # Chuyển thành phút và giây để dễ đọc
        minutes, seconds = divmod(int(total_seconds), 60)
        
        st.markdown(f"---")
        st.markdown(f"⏳ **Thời gian xử lý dự kiến:** Khoảng **{minutes} phút {seconds} giây** cho `{num_products}` sản phẩm (với batch size là `{batch_size}`).")
        st.caption("Thời gian thực tế có thể thay đổi tùy vào tải của API và độ phức tạp của dữ liệu.")


        if st.button("Bắt đầu Phân tích", type="primary", use_container_width=True):
            model = genai.GenerativeModel(model_name)
            all_products = df.to_dict('records')
            
            final_results = []
            
            progress_bar = st.progress(0, text="Đang chuẩn bị...")
            
            batches = [all_products[i:i + batch_size] for i in range(0, len(all_products), batch_size)]
            
            for i, batch in enumerate(batches):
                status_text = f"Đang xử lý lô {i+1}/{len(batches)}... Vui lòng chờ."
                progress_bar.progress((i) / len(batches), text=status_text)
                
                if not batch:
                    continue

                analysis_result = analyze_product_batch_api(batch, model, prompt_text)
                time.sleep(2)

                if analysis_result and 'product_analyses' in analysis_result:
                    batch_map = {row.get('ASIN'): row for row in batch if row.get('ASIN')}
                    for analysis in analysis_result['product_analyses']:
                        asin = analysis.get('asin')
                        if asin in batch_map:
                            original_row = batch_map[asin]
                            original_row.update(analysis)
                            if isinstance(original_row.get('risks'), list):
                                original_row['risks'] = ', '.join(original_row['risks'])
                            final_results.append(original_row)
                else:
                    st.warning(f"Lô {i+1} xử lý thất bại hoặc không có kết quả.")
                    for row in batch:
                        row['classification'] = 'API_ERROR'
                        final_results.append(row)

            progress_bar.progress(1.0, text="Phân tích hoàn tất!")
            
            result_df = pd.DataFrame(final_results)
            
            st.subheader("Kết quả Phân tích")
            st.dataframe(result_df)
            
            csv_output = result_df.to_csv(index=False, encoding='utf-8-sig')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{timestamp}_analyzed_{uploaded_file.name}"

            st.download_button(
                label="📥 Tải về file kết quả",
                data=csv_output,
                file_name=output_filename,
                mime='text/csv',
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"Đã có lỗi xảy ra khi đọc hoặc xử lý file: {e}")