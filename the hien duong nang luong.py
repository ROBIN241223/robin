import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_energy_lines(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    energy_vertical = np.abs(np.diff(gray, axis=0))
    energy_horizontal = np.abs(np.diff(gray, axis=1))
    for i in range(energy_vertical.shape[1]):
        col_energy = energy_vertical[:, i]
        low_energy_indices = np.where(col_energy < np.percentile(col_energy, 25))[0]
        high_energy_indices = np.where(col_energy > np.percentile(col_energy, 75))[0]
        for idx in low_energy_indices:
            cv2.line(img, (i, idx), (i, idx + 1), (255, 0, 0), 1)
        for idx in high_energy_indices:
            cv2.line(img, (i, idx), (i, idx + 1), (0, 0, 255), 1)
    for i in range(energy_horizontal.shape[0]):
        row_energy = energy_horizontal[i, :]
        low_energy_indices = np.where(row_energy < np.percentile(row_energy, 25))[0]
        high_energy_indices = np.where(row_energy > np.percentile(row_energy, 75))[0]
        for idx in low_energy_indices:
            cv2.line(img, (idx, i), (idx + 1, i), (255, 0, 0), 1)
        for idx in high_energy_indices:
            cv2.line(img, (idx, i), (idx + 1, i), (0, 0, 255), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Đường dẫn đến ảnh của bạn
image_path = r'E:\Sounds\pythonProject\2.jpg'
image_with_lines = visualize_energy_lines(image_path)
original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# Tạo subplot để hiển thị ảnh gốc và ảnh đã xử lý theo hàng
plt.figure(figsize=(10, 5))  # Điều chỉnh kích thước hình

# Hiển thị ảnh gốc
plt.subplot(1, 2, 1)  # 1 hàng, 2 cột, vị trí 1
plt.imshow(original_image)
plt.title("Ảnh gốc")

# Hiển thị ảnh đã xử lý
plt.subplot(1, 2, 2)  # 1 hàng, 2 cột, vị trí 2
plt.imshow(image_with_lines)
plt.title("Ảnh với đường năng lượng")

plt.show()