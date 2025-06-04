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
        else:
            print("Không nhận được phản hồi từ Arduino.")


    # Gửi tọa độ hoặc lệnh đặc biệt đến Arduino
    while True:
        try:
            print("\nNhập lệnh:")
            print("1. Nhập tọa độ x, y, z")
            print("2. Gửi lệnh HOME (về vị trí gốc)")
            print("3. Thoát")
            choice = input("Lựa chọn: ")

            if choice == "1":
                # Nhập tọa độ x, y, z từ người dùng
                x = input("Nhập tọa độ x: ")
                y = input("Nhập tọa độ y: ")
                z = input("Nhập tọa độ z: ")

                # Gửi lệnh tọa độ
                command = f"{x},{y},{z}\n"
                send_command(command)

            elif choice == "2":
                # Gửi lệnh HOME
                send_command("HOME\n")

            elif choice == "3":
                print("Thoát chương trình.")
                break

            else:
                print("Lựa chọn không hợp lệ.")

        except ValueError as e:
            print(f"Lỗi nhập liệu: {e}")

except serial.SerialException as e:
    print(f"Không thể kết nối với Arduino: {e}")

finally:
    # Đóng kết nối
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
        print("Đã đóng kết nối với Arduino.")