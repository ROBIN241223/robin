import cv2
import numpy as np

image = cv2.imread('mach.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (int(image.shape[1] / 2.0), int(image.shape[0] / 2.0)))
cv2.imwrite('resized_image.jpg', image)

def apply_filter(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'edge').astype(np.float32)
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            roi = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(roi * kernel)
    return result.astype(np.uint8)

laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
blurred = cv2.GaussianBlur(image, (5, 5), 0)
unsharp_mask = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
laplacian_filtered = apply_filter(image, laplacian_kernel)

cv2.imshow('Original', image)
cv2.imshow('Laplacian', laplacian_filtered)
cv2.imshow('Unsharp Masking', unsharp_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()