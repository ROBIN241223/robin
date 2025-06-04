import threading
import time
import keyboard

from lidar_lib.lidar_scan import LidarScanner

# Khởi tạo đối tượng scanner
# Đảm bảo cổng COM và các tham số là chính xác
scanner = LidarScanner(port="COM6", theta_x=90, theta_y=0, neighbor_range=10, angle_resolution=0.25, pwm=800)

# Chạy start_scan trong một luồng riêng
scan_thread = threading.Thread(target=scanner.start_scan, daemon=True)
scan_thread.start()

# Chờ Lidar khởi động và có dữ liệu
time.sleep(2)

print("Nhấn 'q' để thoát chương trình.")
print("Nhấn Ctrl+C cũng sẽ dừng, nhưng dùng 'q' là tốt nhất để dừng gọn gàng.")

try:
    # Vòng lặp chính để lấy và hiển thị dữ liệu
    while True:
        # Kiểm tra nếu phím 'q' được nhấn để thoát chương trình
        if keyboard.is_pressed("q"):
            print("Dừng chương trình theo yêu cầu 'q'...")
            break  # Thoát vòng lặp chính

        # Lấy dữ liệu từ Lidar
        # Đảm bảo phương thức get_data() trả về 3 giá trị
        distance_x, distance_y, frequency = scanner.get_data()

        # In dữ liệu ra màn hình
        print(f"Distance X: {distance_x} mm, Distance Y: {distance_y} mm, Frequency: {frequency:.1f} Hz")

        # time.sleep(0.01) # Có thể thêm một khoảng dừng nhỏ nếu cần

except KeyboardInterrupt:
    # Xử lý khi nhấn Ctrl+C
    print("\nĐã nhận tín hiệu Ctrl+C. Đang dừng chương trình...")

finally:
    # Khối này luôn chạy khi thoát khỏi try hoặc except
    # Đảm bảo gọi hàm dừng quét của scanner để cleanup tài nguyên
    try:
        scanner.stop_scan()
        print("Đã gọi scanner.stop_scan().")
    except Exception as e:
        print(f"Lỗi khi gọi scanner.stop_scan(): {e}")

# Chương trình kết thúc sau khi thoát vòng lặp hoặc xử lý ngoại lệ
print("Chương trình đã dừng hoàn toàn.")