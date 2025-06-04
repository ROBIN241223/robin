import serial
import time
from flask import Flask, render_template_string, redirect, url_for, flash

# --- Cấu hình ---
SERIAL_PORT = 'COM9'  # <<< !!! THAY ĐỔI THÀNH CỔNG COM ĐÚNG CỦA ARDUINO !!!
BAUD_RATE = 9600
# --- ---

# Khởi tạo Flask app
app = Flask(__name__)
app.secret_key = 'khoabimat123' # Cần thiết cho flash messages

# Khởi tạo kết nối Serial (đặt trong try...except để xử lý lỗi nếu không mở được cổng)
ser = None
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Da ket noi voi Arduino tren cong {SERIAL_PORT}")
    time.sleep(2) # Chờ Arduino khởi động lại sau khi mở cổng Serial
except serial.SerialException as e:
    print(f"Loi: Khong the mo cong {SERIAL_PORT}. Chi tiet: {e}")
    print("Chuong trinh se chay ma khong gui lenh den Arduino.")
except Exception as e:
    print(f"Loi khong xac dinh khi mo cong Serial: {e}")

# HTML Template cho trang web (dùng inline cho đơn giản)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dieu khien LED Arduino</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; background-color: #f4f4f4; }
        h1 { color: #333; }
        .button {
            background-color: #4CAF50; /* Green */
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px;
            cursor: pointer;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }
        .button-off { background-color: #f44336; } /* Red */
        .button:hover { opacity: 0.8; }
        .flash-message {
            padding: 10px;
            margin: 15px auto;
            border-radius: 5px;
            width: 80%;
            max-width: 400px;
            font-weight: bold;
        }
        .flash-success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .flash-error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
</head>
<body>
    <h1>Dieu khien LED Arduino (Chan 13)</h1>

    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for category, message in messages %}
          <div class="flash-message flash-{{ category }}">{{ message }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}

    <div>
        <a href="{{ url_for('control_led', state='on') }}" class="button">BAT LED</a>
        <a href="{{ url_for('control_led', state='off') }}" class="button button-off">TAT LED</a>
    </div>

</body>
</html>
"""

# Route cho trang chính (hiển thị HTML)
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# Route để xử lý lệnh bật/tắt LED
@app.route('/led/<state>')
def control_led(state):
    global ser
    command = ''
    if state.lower() == 'on':
        command = 'H'
        action_text = "BAT"
    elif state.lower() == 'off':
        command = 'L'
        action_text = "TAT"
    else:
        flash("Lenh khong hop le!", "error")
        return redirect(url_for('index'))

    # Gửi lệnh qua Serial nếu kết nối thành công
    if ser and ser.is_open:
        try:
            ser.write(command.encode()) # Gửi ký tự dạng byte
            print(f"Da gui lenh: {command} ({action_text})")
            flash(f"Da gui lenh {action_text} LED!", "success")
        except Exception as e:
            print(f"Loi khi gui lenh qua Serial: {e}")
            flash("Loi khi gui lenh den Arduino!", "error")
    else:
        print("Loi: Ket noi Serial khong san sang.")
        flash("Khong the ket noi voi Arduino!", "error")

    # Chuyển hướng về trang chính sau khi gửi lệnh
    return redirect(url_for('index'))

# Chạy Flask web server
if __name__ == '__main__':
    # Chạy trên địa chỉ 0.0.0.0 để có thể truy cập từ máy khác trong cùng mạng
    # port=5000 là cổng mặc định của Flask
    app.run(host='0.0.0.0', port=5000, debug=True)
    # Đóng cổng Serial khi server dừng (trong trường hợp debug=False hoặc dùng server production)
    if ser and ser.is_open:
        ser.close()
        print("Da dong cong Serial.")