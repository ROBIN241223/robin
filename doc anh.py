import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('2.jpg')

if img is None:
    print("Lỗi: Không thể đọc ảnh. Kiểm tra lại đường dẫn và tên tệp.")
    exit()

height, width, channels = img.shape
print(f"Kích thước ảnh: Rộng {width}px, Cao {height}px, Số kênh màu: {channels}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

blue_channel = img[:, :, 0]
green_channel = img[:, :, 1]
red_channel = img[:, :, 2]

img_gbr = cv2.merge([green_channel, blue_channel, red_channel])

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

normalized_gray = gray_image / 255.0

thresholded_image_low = np.where(normalized_gray < 0.1, 1, 0)
thresholded_image_mid = np.where((normalized_gray >= 0.1) & (normalized_gray <= 0.3), 1, 0)
thresholded_image_high = np.where(normalized_gray > 0.3, 1, 0)

plt.figure(figsize=(20, 10))

plt.subplot(2, 5, 1)
plt.imshow(img_rgb)
plt.title(f'Ảnh Gốc (RGB) - {width}x{height}')
plt.axis('off')

plt.subplot(2, 5, 2)
plt.imshow(blue_channel, cmap='Blues')
plt.title('Kênh Xanh Dương')
plt.axis('off')

plt.subplot(2, 5, 3)
plt.imshow(green_channel, cmap='Greens')
plt.title('Kênh Xanh Lá')
plt.axis('off')

plt.subplot(2, 5, 4)
plt.imshow(red_channel, cmap='Reds')
plt.title('Kênh Đỏ')
plt.axis('off')

plt.subplot(2, 5, 5)
plt.imshow(img_gbr)
plt.title('Ảnh GBR')
plt.axis('off')

plt.subplot(2, 5, 6)
plt.imshow(gray_image, cmap='gray')
plt.title('Ảnh Grayscale')
plt.axis('off')

plt.subplot(2, 5, 7)
plt.imshow(thresholded_image_low, cmap='gray')
plt.title('Ảnh Đen Trắng ( 0.1)')
plt.axis('off')

plt.subplot(2, 5, 8)
plt.imshow(thresholded_image_mid, cmap='gray')
plt.title('Ảnh Đen Trắng (0.2)')
plt.axis('off')

plt.subplot(2, 5, 9)
plt.imshow(thresholded_image_high, cmap='gray')
plt.title('Ảnh Đen Trắng ( 0.3)')
plt.axis('off')


plt.tight_layout()
plt.show()