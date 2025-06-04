from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model (chọn model phù hợp, 'n' là nhanh nhất)
model = YOLO('yolov8n.pt')  # Hoặc 'yolov8s.pt', 'yolov8m.pt', v.v.

# Sử dụng webcam (camera ID 0 là webcam mặc định)
cap = cv2.VideoCapture(0)

# Kiểm tra xem webcam có mở được không
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Cài đặt độ phân giải (tùy chọn, có thể cải thiện hiệu suất)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:  # Vòng lặp vô hạn để xử lý liên tục từ webcam
    ret, frame = cap.read()  # Đọc frame từ webcam

    if not ret:
        print("Error: Could not read frame.")
        break

    # Chạy YOLOv8 để nhận diện
    results = model(frame)

    # Xử lý kết quả (vẽ bounding box, hiển thị label)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            class_id = int(box.cls[0])
            # Class ID 2 thường là xe ô tô, 7 là xe tải (trong COCO dataset)
            if class_id == 2 or class_id == 7:
                confidence = box.conf[0]
                if confidence > 0.3:  # Ngưỡng tin cậy (điều chỉnh cho phù hợp)
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    # Vẽ bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Lấy tên lớp và hiển thị
                    class_name = result.names[class_id] # Lấy tên lớp từ model
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hiển thị frame đã xử lý
    cv2.imshow('Webcam Car Detection', frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()