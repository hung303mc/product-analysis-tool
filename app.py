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
import numpy as np
import logging # <<< DEBUG LOG >>> Thêm thư viện logging

# <<< DEBUG LOG >>> Cấu hình logging để in ra console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


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
Mày là một nhà chiến lược tìm kiếm cơ hội trong thương mại điện tử. Mục tiêu chính của mày là xác định các sản phẩm "winner" MỚI cho một doanh nghiệp online linh hoạt, phù hợp với chiến lược tăng trưởng 2 giai đoạn.

BỐI CẢNH & QUY TẮC KINH DOANH:
1.  **Mô hình Logistics:** - Giai đoạn 1 (Test): Bán hàng theo hình thức Dropshipping. Sản phẩm được ship lẻ trực tiếp từ nhà cung cấp (cụ thể là từ Trung Quốc) đến khách hàng ở Mỹ để kiểm chứng nhu cầu thị trường mà không cần vốn nhập hàng.
- Giai đoạn 2 (Scale): Khi một sản phẩm cho thấy tín hiệu tốt (doanh số, phản hồi tích cực), sẽ tiến hành nhập một lô hàng nhỏ (100-500 sản phẩm) về kho tại Mỹ để tối ưu tốc độ giao hàng (FBM hoặc FBA) và tăng biên lợi nhuận.
2.  **GIỚI HẠN CÂN NẶNG NGHIÊM NGẶT:** Sản phẩm > 4 lbs (~1.8 kg) sẽ tự động bị 'Bỏ qua' (Skip). Lý tưởng là dưới 3 lbs (~1.4 kg).
3.  **Lợi nhuận là trên hết:** Sử dụng chi phí vận chuyển tham khảo (~$15 cho 1lb, ~$25 cho 2lbs, v.v.) để ước tính lợi nhuận ròng tiềm năng.
4.  **THÀNH CÔNG BAN ĐẦU (để tham khảo, không phải giới hạn):** Doanh nghiệp đã có thành công bước đầu với ngách "Các thiết bị cho thời tiết nóng" và "Phụ kiện thời trang mùa hè". Hãy dùng thông tin này làm ngữ cảnh để nhận diện các mô thức thành công (ví dụ: sản phẩm độc đáo giải quyết vấn đề), nhưng TUYỆT ĐỐI KHÔNG chấm điểm thấp cho các sản phẩm thuộc ngành hàng mới. Mục tiêu chính là tìm ra NGÁCH THÀNH CÔNG TIẾP THEO.
5.  **Mục tiêu cốt lõi:** Tìm kiếm các sản phẩm độc đáo, có tiềm năng viral, có thể tìm nguồn từ 1688 trung quốc, Vietnam, ok cho sản phẩm cá nhân hóa
6.  **Tư duy về Review Thấp: Cơ hội trong Rủi ro.** không tự động 'Bỏ qua' một sản phẩm chỉ vì nó có review thấp. Thay vào đó, hãy xem đây là một **CƠ HỘI LỚN**. Review thấp nhưng doanh số vẫn cao cho thấy nhu cầu thị trường rất mạnh và các đối thủ hiện tại đang làm không tốt.
7.  **Xử lý Dữ liệu Thiếu (Đặc biệt là Cân nặng - Weight):** KHÔNG được tự ý ước tính hoặc bịa ra số liệu nếu cột đó trong file CSV bị trống. Trong trường hợp này, hãy tập trung phân tích các yếu tố khác như tiềm năng viral, độ độc đáo, nhu cầu thị trường (dựa trên doanh số/review).


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
    logging.info(f"Bat dau ham analyze_product_batch_api voi {len(product_batch_list)} san pham.") # <<< DEBUG LOG >>>
    if not product_batch_list: return None
    output = io.StringIO()
    fieldnames = product_batch_list[0].keys()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(product_batch_list)
    csv_string = output.getvalue()
    prompt = prompt_template.format(product_batch_csv=csv_string)
    
    # <<< DEBUG LOG >>> In ra prompt để kiểm tra (chỉ vài trăm ký tự đầu)
    logging.info(f"Prompt da tao (200 ky tu dau): {prompt[:200]}...")
    
    try:
        response = model.generate_content(prompt)
        # <<< DEBUG LOG >>> In ra toàn bộ text trả về từ API TRƯỚC khi xử lý
        logging.info(f"API Response Text (RAW): {response.text}")
        
        if response.parts:
            cleaned_text = response.text.strip().replace('```json', '').replace('```', '')
            logging.info(f"Cleaned Text (truoc khi load json): {cleaned_text}") # <<< DEBUG LOG >>>
            return json.loads(cleaned_text)
        
        logging.warning("API response khong co 'parts'.") # <<< DEBUG LOG >>>
        return None
    except Exception as e:
        # <<< DEBUG LOG >>> Ghi log lỗi chi tiết ra console
        logging.error(f"LOI TRONG HAM analyze_product_batch_api: {e}", exc_info=True)
        st.error(f"Lỗi API hoặc JSON: {e}")
        return None

# --- STREAMLIT APP UI ---
def main():
    st.title("🚀 Công Cụ Phân Tích Sản Phẩm")
    st.info("Tải lên file CSV từ Helium 10 để bắt đầu.")

    with st.expander("📖 Xem hướng dẫn sử dụng (Workflow)", expanded=False):
        st.markdown("""
        ...
        """)
    
    # -- Sidebar cài đặt --
    st.sidebar.header("Cài đặt Phân tích")
    model_name = st.sidebar.selectbox(
        "Chọn Model Gemini", 
        ('gemini-2.5-pro', 'gemini-1.5-flash'),
        index=0, 
        help="1.5 Pro cho chất lượng phân tích cao nhất."
    )
    batch_size = st.sidebar.slider("Số sản phẩm mỗi lô", 5, 20, 10)

    st.sidebar.header("Cài đặt Output")
    output_option = st.sidebar.radio("Chọn định dạng output:",
                                     ('Tải File CSV', 'Ghi vào Google Sheet'),
                                     index=1)
                                     
    write_interval = 1
    if output_option == 'Ghi vào Google Sheet':
        gsheet_id = st.sidebar.text_input("ID của Google Sheet 'KHO KẾT QUẢ'", 
                                          "1v0Ms4Mg1L5liXl-5pGRhWoIXSipS0E5z7zGT4G8-JAM")
        write_interval = st.sidebar.slider("Ghi vào Google Sheet sau mỗi (lô)", 1, 10, 5,
                                           help="Ví dụ: chọn 5 nghĩa là cứ xử lý xong 5 lô thì sẽ ghi kết quả vào Sheet một lần.")

    with st.expander("📝 Chỉnh sửa Prompt (Nâng cao)"):
        prompt_text = st.text_area("Prompt Template:", value=DEFAULT_PROMPT, height=400)

    uploaded_file = st.file_uploader("Tải file CSV từ Helium 10 lên đây:", type=["csv"])

    if uploaded_file is not None:
        try:
            logging.info(f"Bat dau doc file: {uploaded_file.name}") # <<< DEBUG LOG >>>
            df = pd.read_csv(uploaded_file, low_memory=False)
            logging.info(f"Doc file thanh cong. Shape ban dau: {df.shape}") # <<< DEBUG LOG >>>

            cols_to_drop = [
                'BSR', 
                'UPC', 
                'GTIN', 
                'EAN', 
                'ISBN',
                'Price Trend (90 days) (%)',
                'ASIN Sales',
                'Sales Trend (90 days) (%)',
                'Last Year Sales',
                'Sales Year Over Year (%)',
                'Length',
                'Width',
                'Height',
                'Storage Fee (Jan - Sep)',
                'Storage Fee (Oct - Dec)',
                'Best Sales Period',
                'Sales to Reviews'
            ]
            for col in cols_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])
            logging.info(f"Da xoa cac cot thua. Shape sau khi xoa: {df.shape}") # <<< DEBUG LOG >>>

            if 'Weight' in df.columns:
                df['Weight'] = df['Weight'].replace('-', np.nan)
                df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
            
            numeric_cols = ['Price', 'Parent Level Sales', 'Revenue', 'Sales', 'BSR']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df = df.fillna('')
            logging.info("Da don dep va dien vao cac o trong.") # <<< DEBUG LOG >>>

            st.success(f"Đã tải lên thành công file '{uploaded_file.name}' với {len(df)} sản phẩm.")
            st.dataframe(df.head())
            
            num_products = len(df)
            time_per_batch_seconds = 45 
            num_batches = (num_products + batch_size - 1) // batch_size
            total_seconds = num_batches * time_per_batch_seconds
            minutes, seconds = divmod(int(total_seconds), 60)
            
            st.markdown(f"---")
            st.markdown(f"⏳ **Thời gian xử lý dự kiến:** Khoảng **{minutes} phút {seconds} giây** cho `{num_products}` sản phẩm (`{num_batches}` lô).")
            st.caption("Thời gian thực tế có thể thay đổi tùy vào tải của API.")

            if st.button("🚀 Bắt đầu Phân tích", type="primary", use_container_width=True):
                logging.info("======= NHAN NUT BAT DAU PHAN TICH =======") # <<< DEBUG LOG >>>
                if output_option == 'Ghi vào Google Sheet' and (not gsheet_id):
                    st.error("Lỗi: Vui lòng nhập ID của Google Sheet 'KHO KẾT QUẢ' vào thanh cài đặt bên trái.")
                    st.stop()
                
                st.session_state.start_time = time.time()
                progress_bar = st.progress(0, text="Đang chuẩn bị...")
                timer_placeholder = st.empty()
                timer_placeholder.markdown("⏳ **Thời gian đã chạy:** 00 phút 00 giây")
                
                model = genai.GenerativeModel(model_name)
                all_products = df.to_dict('records')
                final_results_for_df = []
                batches = [all_products[i:i + batch_size] for i in range(0, len(all_products), batch_size)]
                logging.info(f"Da chia {num_products} san pham thanh {len(batches)} lo.") # <<< DEBUG LOG >>>
                
                OUTPUT_COLUMN_ORDER = [
                    'ASIN', 'URL', 'Image URL', 'Title', 'Price', 'Parent Level Sales', 'Weight',
                    'classification', 'reason', 'viral_potential', 'uniqueness_score', 
                    'market_trend_score', 'logistics_fit', 'estimated_shipping_cost', 
                    'profit_potential', 'risks'
                ]
                
                INPUT_COLUMNS_FOR_AI = [
                    'ASIN', 'Title', 'Price', 'Parent Level Sales', 'Brand', 'Seller', 
                    'Seller Country/Region', 'Weight', 'Age (Month)', 'Review Count', 
                    'Reviews Rating', 'Category', 'Subcategory', 'Number of Images', 'Variation Count'
                ]
                
                worksheet = None
                next_row_to_write = 2 

                # <<< START: LOGIC MỚI - CHUẨN BỊ GOOGLE SHEET TRƯỚC VÒNG LẶP >>>
                if output_option == 'Ghi vào Google Sheet':
                    with st.spinner("Đang chuẩn bị Google Sheet..."):
                        try:
                            timestamp = datetime.now().strftime("%y%m%d_%H%M")
                            # Rút gọn tên file để không quá dài
                            file_base_name = os.path.splitext(uploaded_file.name)[0][:30] 
                            worksheet_name = f"{timestamp}_{file_base_name}"
                            
                            spreadsheet = gc.open_by_key(gsheet_id)
                            # Sao chép từ sheet Template nếu có, nếu không thì tạo mới
                            try:
                                template_sheet = spreadsheet.worksheet("Template")
                                worksheet = spreadsheet.duplicate_sheet(source_sheet_id=template_sheet.id, new_sheet_name=worksheet_name)
                            except gspread.exceptions.WorksheetNotFound:
                                st.info("Không tìm thấy sheet 'Template'. Sẽ tạo sheet mới.")
                                worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1, cols=len(OUTPUT_COLUMN_ORDER))
                            
                            # Ghi header vào sheet mới
                            # Ghi từ ô A1
                            worksheet.update(range_name='B1', values=[OUTPUT_COLUMN_ORDER]) 
                            st.success(f"Đã tạo worksheet '{worksheet_name}' và ghi header từ ô B1 thành công.")
                        except Exception as e:
                            st.error(f"Lỗi khi chuẩn bị Google Sheet: {e}")
                            st.stop()
                # <<< END: LOGIC MỚI - CHUẨN BỊ GOOGLE SHEET >>>

                batch_results_for_sheet = [] # List để tích trữ kết quả cho mỗi lần ghi vào sheet
                for i, batch in enumerate(batches):
                    logging.info(f"--- Bat dau xu ly lo {i+1}/{len(batches)} ---") # <<< DEBUG LOG >>>
                    
                    elapsed_seconds = time.time() - st.session_state.start_time
                    mins, secs = divmod(int(elapsed_seconds), 60)
                    timer_text = f"⏳ **Thời gian đã chạy:** {mins:02d} phút {secs:02d} giây"
                    timer_placeholder.markdown(timer_text)

                    status_text = f"Đang xử lý lô {i+1}/{len(batches)}... Vui lòng chờ."
                    progress_bar.progress((i) / len(batches), text=status_text)

                    if not batch: 
                        logging.warning(f"Lo {i+1} rong, bo qua.") # <<< DEBUG LOG >>>
                        continue
                        
                    filtered_batch = []
                    for product in batch:
                        # product ở đây là một dict chứa TẤT CẢ thông tin
                        # filtered_product là một dict mới chỉ chứa các key trong INPUT_COLUMNS_FOR_AI
                        filtered_product = {key: product.get(key, '') for key in INPUT_COLUMNS_FOR_AI}
                        filtered_batch.append(filtered_product)
                    
                    # Chỉ gửi cái batch đã được lọc đi phân tích
                    analysis_result = analyze_product_batch_api(filtered_batch, model, prompt_text)
                    time.sleep(2)
                    
                    # <<< DEBUG LOG >>> In ra ket qua phan tich cua lo
                    logging.info(f"Ket qua phan tich cho lo {i+1}: {analysis_result}")
                    
                    processed_batch = []
                    if analysis_result and 'product_analyses' in analysis_result:
                        logging.info(f"Lo {i+1} phan tich thanh cong, bat dau ghep du lieu.") # <<< DEBUG LOG >>>
                        batch_map = {str(row.get('ASIN')): row for row in batch if row.get('ASIN')}
                        for analysis in analysis_result['product_analyses']:
                            asin = str(analysis.get('asin'))
                            if asin in batch_map:
                                original_row = batch_map[asin]
                                original_row.update(analysis)
                                if isinstance(original_row.get('risks'), list):
                                    original_row['risks'] = ', '.join(map(str, original_row['risks']))
                                processed_batch.append(original_row)
                            else:
                                # <<< DEBUG LOG >>> Canh bao neu ASIN tu AI khong co trong batch goc
                                logging.warning(f"ASIN '{asin}' tu AI response khong tim thay trong batch goc.")
                    else:
                        logging.error(f"LOI: Lo {i+1} xu ly that bai hoac khong co ket qua. Gan co 'API_ERROR'.") # <<< DEBUG LOG >>>
                        st.warning(f"Lô {i+1} xử lý thất bại hoặc không có kết quả. Gắn cờ 'API_ERROR'.")
                        for row in batch:
                            row['classification'] = 'API_ERROR'
                            processed_batch.append(row)
                    
                    final_results_for_df.extend(processed_batch)
                    
                    if output_option == 'Ghi vào Google Sheet' and worksheet:
                        batch_results_for_sheet.extend(processed_batch)
                        
                        # Điều kiện để ghi:
                        # 1. Đã xử lý đủ số lô trong `write_interval`
                        # 2. Hoặc đây là lô cuối cùng
                        is_last_batch = (i + 1) == len(batches)
                        if (len(batch_results_for_sheet) > 0) and (((i + 1) % write_interval == 0) or is_last_batch):
                            with st.spinner(f"Đang ghi {len(batch_results_for_sheet)} kết quả vào Google Sheet..."):
                                try:
                                    # Chuyển list of dicts thành list of lists theo đúng thứ tự cột
                                    # Sửa lại cho đúng
                                    rows_to_append = []
                                    for row_dict in batch_results_for_sheet:
                                        row_list = [row_dict.get(col, '') for col in OUTPUT_COLUMN_ORDER]
                                        rows_to_append.append(row_list)                                 
                                    range_to_update = f'B{next_row_to_write}'
                                    worksheet.update(range_name=range_to_update, values=rows_to_append)

                                    next_row_to_write += len(rows_to_append)

                                    st.toast(f"✅ Đã ghi thành công {len(batch_results_for_sheet)} sản phẩm vào Sheet!")
                                    batch_results_for_sheet.clear()
                                except Exception as e:
                                    st.error(f"Lỗi khi ghi batch vào Google Sheet: {e}")
                                    # Giữ lại dữ liệu để thử ghi vào lần sau
                    # <<< END: LOGIC GHI BATCH >>>

                progress_bar.progress(1.0, text="Phân tích AI hoàn tất!")
                logging.info("======= HOAN TAT TOAN BO CAC LO =======") # <<< DEBUG LOG >>>

                if not final_results_for_df:
                    st.warning("Không có kết quả nào được tạo ra. Vui lòng kiểm tra lại file input hoặc cài đặt.")
                    st.stop()
                
                result_df = pd.DataFrame(final_results_for_df)
                final_ordered_columns = [col for col in OUTPUT_COLUMN_ORDER if col in result_df.columns]
                other_columns = [col for col in result_df.columns if col not in final_ordered_columns]
                result_df = result_df[final_ordered_columns + other_columns]
                
                st.subheader("Xem Trước Toàn Bộ Kết Quả Phân Tích")
                st.dataframe(result_df)

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
                
                elif output_option == 'Ghi vào Google Sheet' and worksheet:
                    sheet_url = f"https://docs.google.com/spreadsheets/d/{gsheet_id}/edit#gid={worksheet.id}"
                    st.balloons()
                    st.success("🎉 Ghi dữ liệu vào tab mới thành công!")
                    st.link_button("Mở Google Sheet", url=sheet_url, use_container_width=True)

        except Exception as e:
            # <<< DEBUG LOG >>> Ghi log loi tong the cua qua trinh
            logging.error(f"LOI TONG THE trong luc xu ly file: {e}", exc_info=True)
            st.error(f"Đã có lỗi xảy ra trong quá trình xử lý file: {e}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()