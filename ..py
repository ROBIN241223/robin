import serial
import time

# --- !!! THAY ĐÚNG SỐ COM BẠN THẤY TRONG DEVICE MANAGER VÀO ĐÂY !!! ---
SERIAL_PORT = 'COM9'
BAUD_RATE = 9600
# --- ---

print(f"--- Dang thu mo cong {SERIAL_PORT} ---")
try:
    # Cố gắng mở cổng Serial
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    # Nếu mở thành công
    print(f"!!! THÀNH CÔNG: Da mo duoc cong {SERIAL_PORT} !!!")
    ser.close() # Đóng lại ngay lập tức
    print(f"--- Da dong cong {SERIAL_PORT} ---")

except serial.SerialException as e:
    # Nếu lỗi khi mở cổng
    print(f"*** LỖI: Khong the mo cong {SERIAL_PORT}. ***")
    print(f"Chi tiet loi: {e}") # In chi tiết lỗi
except Exception as e:
    print(f"Loi khac khong mong doi: {e}")

# Giữ cửa sổ mở để xem kết quả
input("\nNhan Enter de thoat...")
