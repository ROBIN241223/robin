import cv2
import numpy as np

def calculate_energy(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.abs(sobel_x) + np.abs(sobel_y)

def find_vertical_seam(energy_map):

    rows, cols = energy_map.shape
    dp = np.copy(energy_map)
    parent = np.zeros_like(dp, dtype=int)

    for i in range(1, rows):
        for j in range(cols):
            min_idx = j
            if j > 0 and dp[i - 1, j - 1] < dp[i - 1, min_idx]:
                min_idx = j - 1
            if j < cols - 1 and dp[i - 1, j + 1] < dp[i - 1, min_idx]:
                min_idx = j + 1
            dp[i, j] += dp[i - 1, min_idx]
            parent[i, j] = min_idx

    seam = [np.argmin(dp[-1])]
    for i in range(rows - 1, 0, -1):
        seam.append(parent[i, seam[-1]])

    return seam[::-1]

def remove_vertical_seam(image, seam):

    rows, cols, _ = image.shape
    output = np.zeros((rows, cols - 1, 3), dtype=np.uint8)
    for i in range(rows):
        output[i] = np.delete(image[i], seam[i], axis=0)
    return output

def find_horizontal_seam(energy_map):

    return find_vertical_seam(energy_map.T)

def remove_horizontal_seam(image, seam):

    return remove_vertical_seam(image.transpose(1, 0, 2), seam).transpose(1, 0, 2)

# Đọc ảnh
image_path = 'E:\\Sounds\\pythonProject\\2.jpg'
original_image = cv2.imread(image_path)
if original_image is None:
    print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn.")
    exit()


height, width = original_image.shape[:2]
if width < 2 or height < 2:
    print("Ảnh quá nhỏ để xử lý seam carving.")
    exit()


if width > 800 or height > 600:
    resized_image = cv2.resize(original_image, (800, 600))
else:
    resized_image = original_image.copy()


image_reduced_width = resized_image.copy()
for i in range(50):
    print(f"Đang xóa seam dọc {i + 1}/50")
    energy_map = calculate_energy(image_reduced_width)
    seam = find_vertical_seam(energy_map)
    image_reduced_width = remove_vertical_seam(image_reduced_width, seam)


image_reduced_height = image_reduced_width.copy()
for i in range(50):
    print(f"Đang xóa seam ngang {i + 1}/50")
    energy_map = calculate_energy(image_reduced_height)
    seam = find_horizontal_seam(energy_map)
    image_reduced_height = remove_horizontal_seam(image_reduced_height, seam)


cv2.imwrite('image_reduced_width.jpg', image_reduced_width)
cv2.imwrite('image_reduced_height.jpg', image_reduced_height)
print("Đã xuất ảnh kết quả.")