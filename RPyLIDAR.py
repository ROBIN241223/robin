import time
import keyboard
import serial # <--- Thêm dòng này để import module serial
import math # Thêm math để sử dụng math.radians

from rplidar import RPLidar

# --- Cấu hình ---
# Thay thế 'COM6' bằng cổng COM thực tế của Lidar của bạn
LIDAR_PORT = 'COM6' # Hoặc cổng COM đúng bạn đã kiểm tra trong Device Manager
# Tốc độ Baudrate chuẩn cho RP Lidar A series thường là 115200 cho Standard Scan, 256000 cho Express Scan.
# Thư viện rplidar-python mặc định dùng Standard Scan, nên 115200 là phổ biến.
# Nếu Lidar của bạn chỉ hoạt động ở 256000 (ví dụ: một số model A3/S1), hãy thử thay đổi.
LIDAR_BAUDRATE = 115200
# Góc mục tiêu để lấy khoảng cách (tính từ phía trước Lidar, 0 độ là phía trước, 90 độ là bên trái, 180 độ là phía sau, 270 độ/ -90 độ là bên phải)
TARGET_ANGLE_DEGREE = 90.0
# Sai số cho phép khi tìm góc mục tiêu
ANGLE_TOLERANCE = 2.0 # Độ


# Hàm tìm điểm gần nhất với góc mục tiêu trong một lần quét
def find_point_near_angle(scan, target_angle, tolerance):
    """
    Tìm điểm trong dữ liệu quét có góc gần nhất với góc mục tiêu.
    Scan là list các tuple (quality, angle, distance).
    Góc trong scan là từ 0 đến 360 độ.
    Distance trong scan là mm (đã được chuẩn hóa bởi rplidar-python).
    """
    closest_point = None
    min_angle_diff = float('inf')

    for quality, angle, distance in scan:
        # Tính độ lệch góc, xử lý trường hợp quanh 0/360 độ
        angle_diff = abs(angle - target_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # Kiểm tra khoảng cách hợp lệ (lớn hơn 0) và độ lệch góc nằm trong tolerance
        if angle_diff < min_angle_diff and angle_diff <= tolerance and distance > 0:
            min_angle_diff = angle_diff
            closest_point = (quality, angle, distance)

    return closest_point

# --- Chương trình chính ---
print(f"Đang kết nối tới Lidar tại cổng {LIDAR_PORT}...")

# Sử dụng try...finally để đảm bảo Lidar dừng và cổng đóng ngay cả khi có lỗi
lidar = None
try:
    # Khởi tạo đối tượng Lidar
    # Thử kết nối, có thể cần vài lần thử nếu cổng đang bận
    try:
        lidar = RPLidar(LIDAR_PORT, LIDAR_BAUDRATE)
    except serial.SerialException as e:
         print(f"Lỗi kết nối serial: {e}")
         print(f"Kiểm tra lại cổng {LIDAR_PORT} và đảm bảo không có chương trình nào khác đang sử dụng.")
         raise # Ném lại lỗi để thoát chương trình

    # Lấy thông tin và trạng thái Lidar
    print("Đã kết nối.")
    try:
        info = lidar.get_info()
        print("Thông tin Lidar:", info)
    except Exception as e:
         print(f"Không thể lấy thông tin Lidar: {e}")
         # Tiếp tục nhưng cẩn thận

    try:
        health = lidar.get_health()
        print("Trạng thái Lidar:", health)

        if health[0] != 0: # Status 0 là OK
            print("\n!!! Lidar đang báo lỗi !!!")
            print(f"Mã lỗi: {health[1]}")
            print("Vui lòng kiểm tra đèn báo trên Lidar và tài liệu hướng dẫn để xử lý lỗi phần cứng/firmware.")
            # Không tiếp tục nếu Lidar có lỗi nghiêm trọng
            raise Exception("Lidar is not healthy") # Ném ra ngoại lệ để nhảy đến finally block

    except Exception as e:
         print(f"Không thể lấy trạng thái sức khỏe Lidar hoặc Lidar không khỏe: {e}")
         # Vẫn cố gắng dừng Lidar trong finally block, nhưng thoát
         raise # Ném lại lỗi để thoát chương trình

    print("\nLidar sẵn sàng. Bắt đầu quét...")
    print(f"Lấy khoảng cách tại góc khoảng {TARGET_ANGLE_DEGREE} độ.")
    print("Nhấn 'q' để dừng.")

    # Bắt đầu quét và lặp qua từng lần quét 360 độ
    # iter_scans() là một generator, mỗi lần yield ra một list các điểm cho một vòng quay
    # min_len=100 đảm bảo chỉ xử lý các lần quét có đủ số điểm
    scan_count = 0
    for scan in lidar.iter_scans(min_len=100):
        scan_count += 1
        # scan là list các điểm (quality, angle, distance)

        # Tìm điểm gần góc mục tiêu
        point_at_target_angle = find_point_near_angle(scan, TARGET_ANGLE_DEGREE, ANGLE_TOLERANCE)

        distance_mm = 0 # Giá trị mặc định nếu không tìm thấy điểm
        angle_found = None

        if point_at_target_angle:
            quality, angle, distance_mm = point_at_target_angle
            angle_found = angle
            # Ở đây ta chỉ lấy khoảng cách tại góc TARGET_ANGLE_DEGREE.
            # Nếu bạn cần Distance X và Distance Y theo ý nghĩa tọa độ Descartes, bạn cần tính từ góc và khoảng cách,
            # hoặc lấy khoảng cách tại các góc 0 và 90 độ tương ứng.
            # Ví dụ đơn giản: Coi khoảng cách tại 90 độ là "khoảng cách theo phương ngang" (Distance X theo nghĩa nào đó)
            print(f"Scan {scan_count}: Góc tìm thấy: {angle_found:.2f}°, Khoảng cách: {distance_mm:.2f} mm")
        else:
             print(f"Scan {scan_count}: Không tìm thấy điểm hợp lệ gần {TARGET_ANGLE_DEGREE}°")

        # Tần số quét có thể ước tính trung bình từ thời gian giữa các scan.
        # rplidar-python không cung cấp tần số dễ dàng trong vòng lặp iter_scans.
        # Nếu bạn thực sự cần tần số, cần thêm logic tính thời gian hoặc tìm cách lấy từ Lidar info.
        # Tạm thời bỏ qua tần số trong output này cho đơn giản.

        # Kiểm tra nếu phím 'q' được nhấn để thoát
        if keyboard.is_pressed('q'):
            print("\nĐang dừng Lidar...")
            break # Thoát vòng lặp quét

except serial.SerialException as e:
    # Lỗi này sẽ được bắt nếu không thể mở cổng ban đầu
    print(f"\nChương trình dừng do lỗi Serial: {e}")

except Exception as e:
    # Bắt các lỗi khác, bao gồm cả lỗi 'Lidar is not healthy' tự tạo
    print(f"\nChương trình dừng do lỗi: {e}")

finally:
    # Đảm bảo Lidar dừng và cổng serial được đóng khi kết thúc hoặc có lỗi
    if lidar:
        try:
            print("Đang dọn dẹp...")
            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()
            print("Đã dừng Lidar và ngắt kết nối.")
        except Exception as e:
            print(f"Lỗi khi cố gắng dừng/ngắt kết nối Lidar: {e}")

print("Chương trình đã kết thúc.")