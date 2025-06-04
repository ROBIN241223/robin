from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')  # Thay đổi đường dẫn

# Đường dẫn đến ảnh hoặc video (hoặc 0 cho webcam)
source = 'E:/Sounds/pythonProject/2.jpg'  # Hoặc 0

if isinstance(source, int): # Nếu là số (webcam) thì mở VideoCapture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

while True: # Vòng lặp chính
    if isinstance(source, int): # Nếu là webcam/video
      ret, frame = cap.read()
      if not ret:
          print("Error: Could not read frame.")
          break # Thoát nếu không đọc được frame

    else: # Nếu là ảnh tĩnh
        frame = cv2.imread(source)
        if frame is None: # Kiểm tra xem có đọc được ảnh không
            print(f"Error: Could not read image at {source}")
            exit()



    results = model(frame)

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            confidence = box.conf[0]
            if confidence > 0.5:  # Ngưỡng tin cậy
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                class_id = int(box.cls[0])
                class_name = result.names[class_id]  # Lấy tên class

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Defect Detection', frame)

    if isinstance(source, int): # Nếu là webcam/video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Thoát vòng lặp nếu nhấn 'q'
    else: #Nếu là ảnh tĩnh
        cv2.waitKey(0) # Thì đợi nhấn phím bất kì
        break # Thoát vòng lặp sau khi hiển thị 1 ảnh
# Giải phóng tài nguyên và đóng cửa sổ (sau khi vòng lặp kết thúc)
if isinstance(source, int):
    cap.release()
cv2.destroyAllWindows()