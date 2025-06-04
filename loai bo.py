import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Các hàm trợ giúp (giữ nguyên) ---
def calculate_length(line):
    x1, y1, x2, y2 = line[0]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_angle(line):
    x1, y1, x2, y2 = line[0]
    return math.atan2(y2 - y1, x2 - x1)

def average_point(line):
    x1, y1, x2, y2 = line[0]
    return ((x1 + x2) / 2, (y1 + y2) / 2)

# --- Các tham số (giữ nguyên từ code user) ---
IMAGE_PATH = 'C:/Users/nguye/Pictures/Screenshots/3.png' # !!! THAY ĐỔI NẾU CẦN

CANNY_LOW_THRESH = 50
CANNY_HIGH_THRESH = 150

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 40
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 10

FILTER_MIN_LENGTH = 40
FILTER_MAX_LENGTH = 800
FILTER_PROXIMITY_DIST_THRESH = 15
FILTER_PROXIMITY_ANGLE_THRESH = np.pi / 18

# --- Tham số MỚI cho Inpainting ---
MASK_LINE_THICKNESS = 5 # Độ dày của đường line vẽ lên mask (nên > 1 để inpaint tốt hơn)
INPAINT_RADIUS = 3      # Bán kính lân cận cho thuật toán inpainting
INPAINT_METHOD = cv2.INPAINT_TELEA # Thuật toán: cv2.INPAINT_TELEA hoặc cv2.INPAINT_NS

# --- Logic chính ---
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Lỗi: Không thể tải ảnh từ đường dẫn: {IMAGE_PATH}")
    exit()

output_image_lines_drawn = image.copy() # Ảnh để vẽ đường thẳng phát hiện được lên
h, w = image.shape[:2]
print(f"Ảnh đã tải: {w}x{h} pixels")

# Tiền xử lý và phát hiện đường thẳng (giữ nguyên)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
print("Đã thực hiện chuyển đổi sang ảnh xám và dò biên Canny.")

lines = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                        minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP)

if lines is None:
    print("Không phát hiện được đường thẳng nào ban đầu.")
    lines = []
else:
     print(f"Phát hiện ban đầu {len(lines)} đoạn thẳng.")

# Lọc đường thẳng (giữ nguyên)
filtered_lines_length = []
# Kiểm tra lại lines phòng trường hợp là None hoặc rỗng
if lines is not None and len(lines) > 0:
    for line in lines:
        # Đảm bảo line hợp lệ trước khi tính toán
        if line is not None and len(line) > 0:
            length = calculate_length(line)
            if FILTER_MIN_LENGTH <= length <= FILTER_MAX_LENGTH:
                filtered_lines_length.append(line)
print(f"Sau khi lọc theo chiều dài: còn {len(filtered_lines_length)} đoạn thẳng.")

final_lines = []
# Lọc theo khoảng cách chỉ thực hiện nếu có đường thẳng sau lọc chiều dài
if filtered_lines_length:
    for i, line1 in enumerate(filtered_lines_length):
        is_too_close_to_existing = False
        mid1 = average_point(line1)
        angle1 = calculate_angle(line1)
        for line2 in final_lines:
            mid2 = average_point(line2)
            angle2 = calculate_angle(line2)
            dist_midpoints = math.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
            angle_diff = abs(angle1 - angle2)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            if dist_midpoints < FILTER_PROXIMITY_DIST_THRESH and angle_diff < FILTER_PROXIMITY_ANGLE_THRESH:
                is_too_close_to_existing = True
                break
        if not is_too_close_to_existing:
            final_lines.append(line1)
print(f"Sau khi lọc các đường quá gần nhau: còn {len(final_lines)} đoạn thẳng.")

# Vẽ các đường thẳng cuối cùng LÊN ảnh output_image_lines_drawn (để so sánh)
if not final_lines:
     print("Không còn đường thẳng nào sau khi lọc.")
else:
    print("Vẽ các đường thẳng cuối cùng lên ảnh (để so sánh)...")
    for line in final_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output_image_lines_drawn, (x1, y1), (x2, y2), (0, 0, 255), 2) # Vẽ màu đỏ, dày 2

# --- TẠO MASK VÀ INPAINTING (PHẦN MỚI) ---
# Tạo mặt nạ đen ban đầu
inpainting_mask_lines = np.zeros(image.shape[:2], dtype=np.uint8)
# Ảnh kết quả sau inpainting, khởi tạo bằng ảnh gốc
inpainted_image_lines = image.copy()

# Chỉ tạo mask và inpaint nếu có đường thẳng trong final_lines
if final_lines:
    print(f"Tạo mask cho các đường thẳng với độ dày {MASK_LINE_THICKNESS}...")
    # Vẽ các đường thẳng cuối cùng lên mặt nạ với độ dày lớn hơn
    for line in final_lines:
        x1, y1, x2, y2 = line[0]
        # Vẽ màu trắng (255) lên mask đen
        cv2.line(inpainting_mask_lines, (x1, y1), (x2, y2), (255), MASK_LINE_THICKNESS)

    print(f"Thực hiện inpainting (vẽ lại ảnh) tại các vùng đường thẳng với bán kính lân cận {INPAINT_RADIUS}...")
    # Sử dụng hàm inpaint
    inpainted_image_lines = cv2.inpaint(image,                  # Ảnh gốc
                                        inpainting_mask_lines, # Mask đánh dấu vùng cần vẽ lại
                                        INPAINT_RADIUS,         # Bán kính lân cận
                                        INPAINT_METHOD)         # Thuật toán
else:
    print("Không có đường thẳng nào để thực hiện inpainting.")

# --- Hiển thị kết quả (THÊM SUBPLOT THỨ 4) ---
plt.figure(figsize=(24, 6)) # Điều chỉnh kích thước để chứa 4 ảnh

# Ảnh 1: Gốc
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Ảnh Gốc')
plt.axis('off')

# Ảnh 2: Biên Canny
plt.subplot(1, 4, 2)
plt.imshow(edges, cmap='gray')
plt.title('Ảnh Biên (Canny)')
plt.axis('off')

# Ảnh 3: Đường thẳng vẽ lên ảnh gốc (kết quả từ code gốc của bạn)
plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(output_image_lines_drawn, cv2.COLOR_BGR2RGB))
plt.title(f'Các Đường Thẳng Phát Hiện ({len(final_lines)})')
plt.axis('off')

# Ảnh 4: Ảnh sau khi đã "loại bỏ" (inpainting) đường thẳng
plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(inpainted_image_lines, cv2.COLOR_BGR2RGB))
plt.title('Ảnh Sau Khi "Loại Bỏ" Đường Thẳng')
plt.axis('off')


plt.tight_layout() # Tự động điều chỉnh bố cục
plt.show()

print("Hoàn thành.")