import streamlit as st
import google.generativeai as genai
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
import csv
import time
import json
import os
from datetime import datetime
import io
import pandas as pd
from streamlit_autorefresh import st_autorefresh


# Tạo một "nhịp tim" để giữ kết nối sống, chạy 2 phút một lần (120 giây)
# Tín hiệu này đủ để báo cho các tầng mạng biết session vẫn hoạt động.
st_autorefresh(interval=2 * 60 * 1000, key="heartbeat_refresher")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Product Analysis Tool", page_icon="🚀", layout="wide")

# --- AUTHENTICATION & SETUP ---
# Thiết lập cho cả Gemini và Google Sheets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    
    GCP_CREDS = st.secrets["gcp_service_account"]
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    creds = Credentials.from_service_account_info(GCP_CREDS, scopes=SCOPES)
    gc = gspread.authorize(creds)

except Exception as e:
    st.error(f"Lỗi cài đặt hoặc xác thực: {e}")
    st.info("Hãy kiểm tra lại file .streamlit/secrets.toml và chắc chắn mày đã cấu hình đủ cả GOOGLE_API_KEY và gcp_service_account.")
    st.stop()

# --- PROMPT TEMPLATE (Giữ nguyên như code gốc của mày) ---
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

# --- CORE FUNCTIONS ---
def analyze_product_batch_api(product_batch_list, model, prompt_template):
    # Hàm này giữ nguyên
    if not product_batch_list: return None
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
        return None
    except Exception as e:
        st.error(f"Lỗi API hoặc JSON: {e}")
        return None

# --- STREAMLIT APP UI ---
def main():
    st.title("🚀 Công Cụ Phân Tích Sản Phẩm")
    st.info("Tải lên file CSV từ Helium 10 để bắt đầu.")

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
    
    # -- Sidebar cài đặt --
    st.sidebar.header("Cài đặt Phân tích")
    model_name = st.sidebar.selectbox(
        "Chọn Model Gemini", 
        ('gemini-2.5-pro', 'gemini-1.5-pro', 'gemini-1.5-flash'),
        index=0, 
        help="2.5 Pro cho chất lượng phân tích cao nhất."
    )
    batch_size = st.sidebar.slider("Số sản phẩm mỗi lô", 5, 20, 10)

    # <<< THÊM LỰA CHỌN OUTPUT MỚI >>>
    st.sidebar.header("Cài đặt Output")
    output_option = st.sidebar.radio("Chọn định dạng output:",
                                     ('Tải File CSV', 'Ghi vào Google Sheet'),
                                     index=1)
                                     
    # Các trường nhập liệu cho Google Sheet chỉ hiện ra khi cần
    gsheet_id = ""
    if output_option == 'Ghi vào Google Sheet':
        gsheet_id = st.sidebar.text_input("ID của Google Sheet 'KHO KẾT QUẢ'", 
                                        "1v0Ms4Mg1L5liXl-5pGRhWoIXSipS0E5z7zGT4G8-JAM")

    with st.expander("📝 Chỉnh sửa Prompt (Nâng cao)"):
        prompt_text = st.text_area("Prompt Template:", value=DEFAULT_PROMPT, height=400)

    uploaded_file = st.file_uploader("Tải file CSV từ Helium 10 lên đây:", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Đã tải lên thành công file '{uploaded_file.name}' với {len(df)} sản phẩm.")
            st.dataframe(df.head())
            
            # Ước tính thời gian
            num_products = len(df)
            time_per_batch_seconds = 60 
            num_batches = (num_products + batch_size - 1) // batch_size
            total_seconds = num_batches * time_per_batch_seconds
            minutes, seconds = divmod(int(total_seconds), 60)
            
            st.markdown(f"---")
            st.markdown(f"⏳ **Thời gian xử lý dự kiến:** Khoảng **{minutes} phút {seconds} giây** cho `{num_products}` sản phẩm.")
            st.caption("Thời gian thực tế có thể thay đổi tùy vào tải của API.")
            
            if st.button("🚀 Bắt đầu Phân tích", type="primary", use_container_width=True):
                if output_option == 'Ghi vào Google Sheet' and (not gsheet_id or gsheet_id == "DÁN_ID_FILE_KHO_KẾT_QUẢ_VÀO_ĐÂY"):
                    st.error("Lỗi: Vui lòng nhập ID của Google Sheet 'KHO KẾT QUẢ' vào thanh cài đặt bên trái.")
                    st.stop()
                
                # --- PHẦN LOGIC XỬ LÝ ĐÃ CÓ THANH TIẾN TRÌNH ---
                progress_bar = st.progress(0, text="Đang chuẩn bị...")
                
                model = genai.GenerativeModel(model_name)
                all_products = df.to_dict('records')
                final_results = []
                batches = [all_products[i:i + batch_size] for i in range(0, len(all_products), batch_size)]
                
                for i, batch in enumerate(batches):
                    # Cập nhật thanh tiến trình và status text
                    status_text = f"Đang xử lý lô {i+1}/{len(batches)}... Vui lòng chờ."
                    progress_bar.progress((i) / len(batches), text=status_text)

                    if not batch: continue
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
                
                progress_bar.progress(1.0, text="Phân tích AI hoàn tất!")

                if not final_results:
                     st.warning("Không có kết quả nào được tạo ra. Vui lòng kiểm tra lại file input hoặc cài đặt.")
                     st.stop()
                
                # 1. Tạo DataFrame ban đầu
                temp_df = pd.DataFrame(final_results)

                # 2. Định nghĩa thứ tự cột mong muốn (copy từ file python cũ)
                OUTPUT_COLUMN_ORDER = [
                    # Core Info
                    'ASIN', 'URL', 'Image URL', 'Title', 'Price', 'Parent Level Sales', 'Weight',
                    # Holistic Analysis
                    'classification', 'reason',
                    # Marketing & Strategy Scores
                    'viral_potential', 'uniqueness_score', 'market_trend_score',
                    # Logistics & Profitability Scores
                    'logistics_fit', 'estimated_shipping_cost', 'profit_potential',
                    # Risks
                    'risks'
                ]

                # 3. Tạo danh sách cột cuối cùng để tránh lỗi KeyError
                # Lấy các cột có trong list order và cũng có trong dataframe
                final_ordered_columns = [col for col in OUTPUT_COLUMN_ORDER if col in temp_df.columns]
                # Lấy các cột còn lại trong dataframe mà không có trong list order (để không làm mất dữ liệu)
                other_columns = [col for col in temp_df.columns if col not in final_ordered_columns]

                # 4. Sắp xếp lại DataFrame
                result_df = temp_df[final_ordered_columns + other_columns]
                
                st.subheader("Xem Trước Kết Quả Phân Tích")
                st.dataframe(result_df)

                # <<< LOGIC HIỂN THỊ OUTPUT DỰA TRÊN LỰA CHỌN >>>
                if output_option == 'Tải File CSV':
                    with st.spinner("Đang chuẩn bị file CSV..."):
                        csv_output = result_df.to_csv(index=False).encode('utf-8-sig')
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = f"{timestamp}_analyzed_{uploaded_file.name}"
                        st.download_button(
                            label="📥 Tải về file kết quả CSV",
                            data=csv_output,
                            file_name=output_filename,
                            mime='text/csv',
                            use_container_width=True
                        )
                
                elif output_option == 'Ghi vào Google Sheet':
                    with st.spinner("Đang tạo tab mới và ghi dữ liệu vào Google Sheet..."):
                        timestamp = datetime.now().strftime("%y%m%d_%H%M")
                        worksheet_name = f"{timestamp}_{os.path.splitext(uploaded_file.name)[0]}"
                        
                        try:
                            spreadsheet = gc.open_by_key(gsheet_id)
                            template_sheet = spreadsheet.worksheet("Template")
                            worksheet = spreadsheet.duplicate_sheet(source_sheet_id=template_sheet.id, new_sheet_name=worksheet_name)
                            
                            set_with_dataframe(worksheet, result_df, row=1, col=2, include_index=False, include_column_header=True, resize=False)
                            
                            sheet_url = f"https://docs.google.com/spreadsheets/d/{gsheet_id}/edit#gid={worksheet.id}"
                            
                            st.balloons()
                            st.success("🎉 Ghi dữ liệu vào tab mới thành công!")
                            st.link_button("Mở Google Sheet", url=sheet_url, use_container_width=True)
                        
                        except gspread.exceptions.WorksheetNotFound:
                            st.error("Lỗi: Không tìm thấy sheet có tên 'Template' trong file Google Sheet của mày. Hãy tạo một sheet và đặt tên chính xác là 'Template'.")
                        except Exception as e:
                            st.error(f"Lỗi khi thao tác với Google Sheet: {e}")

        except Exception as e:
            st.error(f"Đã có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()