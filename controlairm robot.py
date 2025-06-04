from ultralytics import YOLO
import cv2
import numpy as np
import serial
import time

# Kết nối với Arduino qua giao tiếp serial
port = "COM4"  # Thay bằng cổng COM của Arduino của bạn
baud_rate = 9600

try:
    # Tạo đối tượng serial
    arduino = serial.Serial(port, baud_rate, timeout=1)
    print("Connected to Arduino.")


    def send_command(command):
        """Gửi lệnh đến Arduino và nhận phản hồi"""
        arduino.write(command.encode())  # Gửi lệnh
        print(f"Đã gửi lệnh: {command.strip()}")

        time.sleep(1)  # Tạm dừng để Arduino xử lý
        if arduino.in_waiting > 0:
            response = arduino.readline().decode().strip()
            print(f"Phản hồi từ Arduino: {response}")
            return response
        else:
            print("Không nhận được phản hồi từ Arduino.")
            return None


    # Load the trained YOLO model
    model = YOLO('E:/Sounds/pythonProject/2.jpg')  # Thay đổi đường dẫn

    # Đường dẫn đến ảnh hoặc video (hoặc 0 cho webcam)
    source = 'E:/Sounds/pythonProject/2.jpg'  # Hoặc 0

    if isinstance(source, int):  # Nếu là số (webcam) thì mở VideoCapture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

    # Tọa độ cố định trong code
    pickup_position = (100, 200, 50)  # Tọa độ nhặt (x, y, z)
    drop_position = (200, 300, 50)  # Tọa độ thả (x, y, z)
    home_position = (0, 0, 0)  # Tọa độ gốc

    while True:  # Vòng lặp chính
        # Đọc frame từ webcam hoặc tải ảnh tĩnh
        if isinstance(source, int):  # Nếu là webcam/video
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break  # Thoát nếu không đọc được frame
        else:  # Nếu là ảnh tĩnh
            frame = cv2.imread(source)
            if frame is None:  # Kiểm tra xem có đọc được ảnh không
                print(f"Error: Could not read image at {source}")
                exit()

        # Nhận diện đối tượng từ xử lý ảnh
        results = model(frame)
        detected = False  # Cờ để kiểm tra xem có đối tượng nào được phát hiện không

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                confidence = box.conf[0]
                if confidence > 0.5:  # Ngưỡng tin cậy
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]  # Lấy tên class

                    # Vẽ khung và hiển thị thông tin đối tượng
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Gửi lệnh điều khiển tới Arduino
                    detected = True
                    print(f"Đối tượng phát hiện: {class_name}, độ tin cậy: {confidence:.2f}")

        # Hiển thị hình ảnh xử lý
        cv2.imshow('Defect Detection', frame)

        # Điều khiển robot nếu phát hiện đối tượng
        if detected:
            print("Phát hiện đối tượng, bắt đầu điều khiển robot...")

            # Nhặt đối tượng
            send_command(f"{pickup_position[0]},{pickup_position[1]},{pickup_position[2]}\n")
            response = send_command("GRIP\n")  # Lệnh gắp
            if response == "SUCCESS":
                print("Gắp thành công!")

                # Di chuyển đến vị trí thả
                send_command(f"{drop_position[0]},{drop_position[1]},{drop_position[2]}\n")
                response = send_command("RELEASE\n")  # Lệnh thả
                if response == "SUCCESS":
                    print("Thả thành công!")

                    # Quay về vị trí gốc
                    send_command(f"{home_position[0]},{home_position[1]},{home_position[2]}\n")
                else:
                    print("Thả không thành công!")
            else:
                print("Gắp không thành công!")

        # Kiểm tra nếu nhấn 'q' để thoát (cho webcam)
        if isinstance(source, int):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:  # Nếu là ảnh tĩnh
            cv2.waitKey(0)  # Đợi nhấn phím bất kì
            break

    # Giải phóng tài nguyên và đóng cửa sổ
    if isinstance(source, int):
        cap.release()
    cv2.destroyAllWindows()

except serial.SerialException as e:
    print(f"Không thể kết nối với Arduino: {e}")
finally:
    # Đóng kết nối
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
        print("Đã đóng kết nối với Arduino.")