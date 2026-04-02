# 🏠 Đặc tả Luồng Người Dùng (User Flow) - Hệ thống Dự đoán Giá Nhà (CatBoost)

Tài liệu này mô tả chi tiết luồng tương tác của người dùng trên giao diện ứng dụng (Frontend Streamlit) và cách thức nó kết nối, xử lý payload dữ liệu phức tạp trước khi gửi đến các API endpoint của Backend (FastAPI).

---

## 1. Khởi động & Cấu hình Hệ thống (Sidebar)

Khu vực Sidebar được thiết kế để người dùng (thường là Admin hoặc Data Scientist) kiểm tra trạng thái và thiết lập môi trường AI.

* **Kiểm tra trạng thái (Health Check):**
    * **Hệ thống:** Tự động gọi ngầm API `GET /health` ngay khi ứng dụng khởi chạy.
    * **Hiển thị:** Báo cáo trạng thái máy chủ (Ví dụ: 🟢 Server Online) để đảm bảo Backend đã sẵn sàng nhận request.
* **Quản lý & Tải Mô hình (Model Management):**
    * **Hệ thống:** Gọi API `GET /models` để lấy danh sách các mô hình hồi quy (Regression Models) hiện có trong hệ thống (vd: `final_model_Cat`, `XGBoost_v2`).
    * **Thao tác:** Người dùng chọn một mô hình từ danh sách Dropdown và nhấn nút **"Load Mô Hình"**.
    * **Xử lý:** Frontend gọi API `POST /models/load/{model_name}` để ra lệnh cho Backend nạp trọng số mô hình vào bộ nhớ RAM, sẵn sàng cho việc suy luận (inference).

---

## 2. Tương tác & Nghiệp vụ Định giá (Main Area)

Màn hình chính được tổ chức thành **3 Tabs** đáp ứng 3 kịch bản sử dụng (Use Cases) chính của hệ thống định giá bất động sản.

### 📝 Tab 1: Định giá Đơn lẻ (Single Predict)
* **Mục đích:** Tra cứu nhanh giá trị của một căn nhà cụ thể dựa trên các đặc trưng (features) quan trọng.
* **Chiến lược UI (Gom nhóm & Tối giản):** Do Schema đầu vào có tới ~75 trường dữ liệu (Ames Housing), giao diện chỉ hiển thị khoảng 10-15 trường cốt lõi có sức ảnh hưởng lớn nhất đến giá trị, được chia làm 3 nhóm:
    1. **Đánh giá & Vị trí:** Năm xây dựng, Chất lượng tổng thể, Khu vực...
    2. **Diện tích (Yếu tố quyết định):** Diện tích đất, Diện tích sinh hoạt, Tầng hầm...
    3. **Phòng ốc & Tiện ích:** Số phòng ngủ, Số phòng tắm, Garage...
* **Xử lý Dữ liệu ngầm (Payload Construction):**
    * Khi người dùng nhấn **"Dự đoán"**, Frontend sẽ thu thập các giá trị từ Form.
    * **Quan trọng:** Frontend tự động "bơm" (inject) các giá trị mặc định cho ~60 trường dữ liệu phụ còn lại (ví dụ: `PoolArea: 0`, `Condition1: Norm`) để tạo thành một cục JSON Payload hoàn chỉnh khớp chuẩn Schema của Backend.
* **Hệ thống xử lý:** Gửi JSON Payload qua API `POST /predict`.
* **Kết quả đầu ra:** * Hiển thị mức giá dự đoán được định dạng tiền tệ (Ví dụ: `$206,462.79`).
    * Hiển thị độ tin cậy của mô hình (Confidence score) và tên mô hình đang sử dụng.

### 📁 Tab 2: Định giá Hàng loạt (Batch Predict)
* **Mục đích:** Ước tính giá trị cho một tệp danh sách nhiều bất động sản cùng lúc, phục vụ phân tích thị trường hoặc thẩm định giá danh mục đầu tư.
* **Thao tác:** Người dùng tải lên (upload) tệp `.csv` chứa bảng dữ liệu các căn nhà (các cột phải khớp với Schema quy định).
* **Hệ thống xử lý:** Đẩy tệp qua API `POST /predict-batch`.
* **Kết quả đầu ra:** * Hiển thị bảng dữ liệu (Dataframe) bao gồm thông tin gốc kèm thêm một cột mới là "Predicted Price" (Giá dự đoán).
    * Cung cấp nút tải xuống (Download) bảng kết quả dưới dạng CSV để lưu trữ.

### 🔍 Tab 3: Phân tích Chuyên sâu (Predict with Explanation)
* **Mục đích:** Cung cấp tính minh bạch (Explainable AI), giúp người dùng (Nhà môi giới, Khách hàng) hiểu rõ tại sao AI lại đưa ra mức giá đó.
* **Thao tác:** Người dùng nhập thông số căn nhà tương tự như Tab 1.
* **Hệ thống xử lý:** Gửi dữ liệu qua API `POST /predict-explain`.
* **Kết quả đầu ra:** * Hiển thị mức giá ước tính.
    * **Trực quan hóa (Visualization):** Vẽ biểu đồ (Ví dụ: Bar chart thể hiện Feature Importance hoặc Waterfall chart từ thư viện SHAP) minh họa rõ ràng mức độ đóng góp của từng yếu tố. 
    * *Ví dụ: Biểu đồ cho thấy Diện tích 1710 sqft làm TĂNG giá trị thêm $50,000, nhưng Năm xây dựng cũ (2003) làm GIẢM giá trị đi $5,000.*

---

## 3. Các lưu ý Kỹ thuật (Technical Notes)

* **State Management:** Sử dụng `st.session_state` để lưu trữ mô hình hiện tại đang được active, tránh việc gọi lại API `/health` hoặc `/models` không cần thiết mỗi khi luồng UI render lại.
* **Error Handling:** Tất cả các lệnh gọi API (`requests.post/get`) đều được bọc trong khối `try-except` để bắt lỗi mất kết nối Backend hoặc lỗi HTTP 500, đảm bảo giao diện hiển thị thông báo thân thiện (Alert) thay vì crash ứng dụng.
* **Formatting:** Dữ liệu trả về từ Backend (dạng float nhiều số thập phân) luôn được format lại ở Frontend theo chuẩn tiền tệ (có dấu phẩy ngăn cách hàng nghìn) trước khi hiển thị cho end-user.