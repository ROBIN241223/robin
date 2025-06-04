import cv2
from ultralytics import YOLO
import serial
import time

# Kết nối với Arduino qua cổng Serial (thay 'COM7' bằng cổng Serial của bạn)
# Hãy đảm bảo cổng Serial này đang mở và hoạt động
try:
    arduino = serial.Serial(port='COM7', baudrate=9600, timeout=1)
    time.sleep(2) # Chờ kết nối Serial được thiết lập
    print("Kết nối Serial tới Arduino thành công.")
except serial.SerialException as e:
    print(f"Lỗi kết nối Serial tới COM7: {e}")
    print("Vui lòng kiểm tra cổng COM và kết nối Arduino.")
    arduino = None # Đặt arduino thành None nếu kết nối thất bại


def send_command_to_arduino(command):
    """Gửi lệnh tới Arduino"""
    if arduino is not None and arduino.isOpen():
        try:
            arduino.write(command.encode())
            # print(f"Đã gửi lệnh '{command}' tới Arduino.")
            time.sleep(0.1) # Giảm thời gian chờ nếu cần
        except serial.SerialException as e:
            print(f"Lỗi khi gửi lệnh tới Arduino: {e}")
    else:
        # print("Không thể gửi lệnh: Kết nối Arduino không khả dụng.")
        pass # Không in lỗi liên tục nếu arduino là None


def main():
    # Load mô hình YOLOv8 đã huấn luyện
    # --- CHỖ NÀY ĐÃ ĐƯỢC SỬA ---
    # Thay thế 'duong/dan/den/model/cua/ban.pt' bằng đường dẫn THỰC TẾ của tệp model YOLO của bạn (.pt)
    model_path = 'duong/dan/den/model/cua/ban.pt' # VÍ DỤ: 'yolov8n.pt'
    try:
        model = YOLO(model_path)
        print(f"Đã tải model YOLO thành công từ: {model_path}")
    except Exception as e:
        print(f"Lỗi khi tải model từ {model_path}: {e}")
        print("Vui lòng kiểm tra lại đường dẫn tệp model.")
        if arduino is not None and arduino.isOpen():
             arduino.close()
        return # Thoát chương trình nếu không tải được model

    # Kết nối camera
    camera = cv2.VideoCapture(0)  # Camera mặc định (thường là 0)
    if not camera.isOpened():
        print("Không thể mở camera. Kiểm tra kết nối hoặc thay số index camera (ví dụ: 1, 2).")
        if arduino is not None and arduino.isOpen():
             arduino.close()
        return

    print("Bắt đầu kiểm tra bo mạch. Nhấn 'q' để thoát.")

    while True:
        # Đọc khung hình từ camera
        ret, frame = camera.read()
        if not ret:
            print("Không thể đọc dữ liệu từ camera. Thoát chương trình.")
            break

        # Phân tích hình ảnh bằng YOLOv8
        # Truyền khung hình (frame) vào model đã tải
        results = model(frame, conf=0.5) # Thêm ngưỡng tin cậy (confidence threshold), bạn có thể điều chỉnh 0.5

        # Biến cờ để kiểm tra xem có phát hiện "Lỗi" không
        loi_detected = False

        # Hiển thị kết quả và kiểm tra phát hiện "Lỗi"
        for result in results:
            # Kiểm tra xem có bounding box nào được phát hiện không
            if result.boxes is not None:
                 for box in result.boxes:
                     x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ từ tensor
                     confidence = box.conf[0] # Lấy độ tin cậy
                     class_id = int(box.cls[0]) # Lấy ID lớp
                     class_name = model.names[class_id] # Lấy tên lớp từ model

                     # Bạn cần xác định tên lớp hoặc ID lớp nào tương ứng với "Lỗi"
                     # Ví dụ: Nếu lớp 'Loi' có ID là 0, bạn có thể kiểm tra:
                     # if class_name == 'Loi': # Hoặc if class_id == 0:
                     #     loi_detected = True
                     #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Vẽ khung đỏ
                     #     cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                     # --- Hiện tại, code gốc của bạn vẽ "Loi" cho MỌI phát hiện. ---
                     # --- Nếu bạn chỉ muốn vẽ cho lớp "Loi" cụ thể, hãy điều chỉnh đoạn if ở trên. ---
                     # --- Nếu model của bạn chỉ được huấn luyện để phát hiện lỗi, thì code hiện tại có thể tạm chấp nhận. ---

                     # Ví dụ đơn giản: Cứ phát hiện bất cứ gì thì coi là lỗi và vẽ khung
                     # Nếu bạn cần phân loại lỗi, hãy sửa lại logic kiểm tra class_name/class_id ở trên
                     loi_detected = True # Đặt cờ là True nếu có bất kỳ phát hiện nào
                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Vẽ khung đỏ
                     cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        # Hiển thị khung hình với kết quả
        cv2.imshow('Kiểm tra bo mach', frame)

        # Kiểm tra nếu phát hiện lỗi (dựa vào cờ loi_detected)
        if loi_detected:
            print("Phát hiện lỗi trên bo mạch! Gửi tín hiệu tới Arduino.")
            send_command_to_arduino('1')  # Gửi lệnh kích hoạt gạt
        # else:
            # print("Không phát hiện lỗi.") # Có thể bỏ dòng này để tránh in quá nhiều

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    camera.release()
    cv2.destroyAllWindows()
    # Đóng kết nối Serial nếu nó đang mở
    if arduino is not None and arduino.isOpen():
        arduino.close()
        print("Đã đóng kết nối Serial.")

if __name__ == "__main__":
    main()