import cv2
import numpy as np
import random
import os


def show_image(image, title="2.jpg"):
    """Hiển thị ảnh với tiêu đề."""
    cv2.imshow(title, image)
    cv2.waitKey(0)  # Chờ bấm phím bất kỳ để đóng cửa sổ
    cv2.destroyAllWindows()

# 1. Chọn và hiển thị ảnh ngẫu nhiên
def load_random_image(image_folder="."): # '.' là thư mục hiện tại.
    """Chọn ngẫu nhiên một ảnh từ thư mục và trả về."""
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    if not image_files:
        raise ValueError("Không tìm thấy file ảnh nào trong thư mục.")
    random_image_file = random.choice(image_files)
    image_path = os.path.join(image_folder, random_image_file)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Đọc ảnh grayscale
    if img is None:
      raise ValueError(f"Không thể load ảnh ở {image_path}")
    return img
try:
    img = load_random_image()
    show_image(img, "Ảnh Gốc")
except ValueError as e:
    print(e)
    exit()

# 2. Áp dụng filter 1D trung bình (1x5) theo hàng
def filter_1d_horizontal_average(image):
    """Áp dụng filter trung bình 1x5 theo chiều ngang."""
    kernel = np.array([1, 1, 1, 1, 1]) / 5.0  # Tạo kernel (filter)
    filtered_img = cv2.filter2D(image, -1, kernel)  # Dùng filter2D
    return filtered_img

filtered_img_avg_h = filter_1d_horizontal_average(img)
show_image(filtered_img_avg_h, "Filter Trung Bình 1x5 (Hàng)")


# 3. Filter 1D có trọng số (1 2 4 2 1) theo hàng
def filter_1d_horizontal_weighted(image):
    """Áp dụng filter 1x5 có trọng số [1 2 4 2 1] theo chiều ngang."""
    kernel = np.array([1, 2, 4, 2, 1]) / 10.0  # Tổng trọng số = 10
    filtered_img = cv2.filter2D(image, -1, kernel)
    return filtered_img

filtered_img_weighted_h = filter_1d_horizontal_weighted(img)
show_image(filtered_img_weighted_h, "Filter Trọng Số 1x5 (Hàng)")


# 4. Xoay filter và áp dụng theo cột
def filter_1d_vertical_average(image):
    """Áp dụng filter trung bình 1x5 theo chiều dọc."""
    kernel = np.array([[1], [1], [1], [1], [1]]) / 5.0 # Kernel dọc
    filtered_img = cv2.filter2D(image, -1, kernel)
    return filtered_img

def filter_1d_vertical_weighted(image):
    """Áp dụng filter 5x1 có trọng số theo chiều dọc."""
    kernel = np.array([[1], [2], [4], [2], [1]]) / 10.0  # Kernel dọc
    filtered_img = cv2.filter2D(image, -1, kernel)
    return filtered_img

filtered_img_avg_v = filter_1d_vertical_average(img)
show_image(filtered_img_avg_v, "Filter Trung Bình 5x1 (Cột)")

filtered_img_weighted_v = filter_1d_vertical_weighted(img)
show_image(filtered_img_weighted_v, "Filter Trọng Số 5x1 (Cột)")

# 5. Filter 2D (3x3)

# 5.1 Filter trung bình 3x3
def filter_2d_average(image):
    """Áp dụng filter trung bình 3x3."""
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0  # Tạo kernel 3x3
    filtered_img = cv2.filter2D(image, -1, kernel)
    return filtered_img

filtered_img_2d_avg = filter_2d_average(img)
show_image(filtered_img_2d_avg, "Filter Trung Bình 3x3")

# 5.2 Filter 2D có trọng số (1 2 1; 2 8 2; 1 2 1)
def filter_2d_weighted(image):
    """Áp dụng filter 3x3 có trọng số."""
    kernel = np.array([[1, 2, 1],
                       [2, 8, 2],
                       [1, 2, 1]], dtype=np.float32) / 20.0
    filtered_img = cv2.filter2D(image, -1, kernel)
    return filtered_img

filtered_img_2d_weighted = filter_2d_weighted(img)
show_image(filtered_img_2d_weighted, "Filter Trọng Số 3x3")