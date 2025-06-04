from ultralytics import YOLO
import cv2
import numpy as np

# Load mô hình đã huấn luyện (thay đổi đường dẫn nếu cần)
model = YOLO('path/to/your/best.pt')  # Đường dẫn tuyệt đối hoặc tương đối

# Nguồn đầu vào: ảnh, video, hoặc webcam
source = 'path/to/your/image.jpg'  # Thay bằng đường dẫn ảnh của bạn
# source = 'path/to/your/video.mp4'  # Hoặc đường dẫn video
# source = 0  # Hoặc 0 để dùng webcam

# Nếu là webcam hoặc video, mở VideoCapture
if isinstance(source, int) or (isinstance(source, str) and source.endswith(('.mp4', '.avi', '.mov'))):  # Kiểm tra cả số (webcam) và đuôi file video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        exit()

while True:  # Vòng lặp chính
    if isinstance(source, int) or (isinstance(source, str) and source.endswith(('.mp4', '.avi', '.mov'))): # Nếu là webcam/video
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break  # Thoát vòng lặp nếu hết video hoặc lỗi
    else: # Nếu là ảnh tĩnh
        frame = cv2.imread(source)
        if frame is None:
            print(f"Error: Could not read image at {source}")
            exit()


    # Chạy YOLOv8 trên frame
    results = model(frame)

    # Xử lý kết quả và vẽ bounding box
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            confidence = box.conf[0]
            if confidence > 0.5:  # Ngưỡng tin cậy (điều chỉnh)
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hiển thị kết quả
    cv2.imshow('Defect Detection', frame)

    # Xử lý sự kiện bàn phím
    if isinstance(source, int) or (isinstance(source, str) and source.endswith(('.mp4', '.avi', '.mov'))):  # Webcam/video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Thoát nếu nhấn 'q'
    else:  # Ảnh tĩnh
        cv2.waitKey(0)  # Đợi phím bất kỳ
        break #Thoát sau khi hiển thị ảnh

# Giải phóng tài nguyên và đóng cửa sổ
if isinstance(source, int) or (isinstance(source, str) and source.endswith(('.mp4', '.avi', '.mov'))):
    cap.release()
cv2.destroyAllWindows()