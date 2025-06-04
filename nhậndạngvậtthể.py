import cv2

# 1. Tải bộ phân loại Haar Cascade cho khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Đọc ảnh (hoặc lấy khung hình từ video)
img = cv2.imread('2.jpg')  # Thay bằng đường dẫn ảnh của bạn
# Hoặc, nếu dùng webcam:
# cap = cv2.VideoCapture(0)
# ret, img = cap.read()

# 3. Chuyển ảnh sang ảnh xám (grayscale)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. Phát hiện khuôn mặt
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 5. Vẽ bounding box xung quanh khuôn mặt
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 6. Hiển thị kết quả
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Nếu dùng webcam, thêm vòng lặp while và giải phóng webcam khi xong:
# while True:
#     ret, img = cap.read()
#     # ... (các bước 3, 4, 5) ...
#     cv2.imshow('Detected Faces', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
#         break
# cap.release()
# cv2.destroyAllWindows()