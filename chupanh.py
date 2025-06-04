import cv2
import face_recognition
import tkinter as tk
from tkinter import Entry

# Danh sách mã hóa gương mặt đã đăng ký
known_face_encodings = []
known_face_names = []

# Cách thêm gương mặt đã biết (ví dụ)
def load_known_faces():
    # Thêm gương mặt đã biết
    image = face_recognition.load_image_file("known_face.jpg")  # Đường dẫn tệp ảnh gương mặt đã biết
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append("User")

# Hàm hiển thị bàn phím ảo
def show_virtual_keyboard():
    def on_key_press(key):
        current_text = entry.get()
        if key == "Space":
            entry.insert(tk.END, " ")
        elif key == "Backspace":
            entry.delete(len(current_text)-1, tk.END)
        else:
            entry.insert(tk.END, key)

    # Tạo cửa sổ bàn phím ảo
    keyboard_window = tk.Tk()
    keyboard_window.title("Bàn phím ảo")
    keyboard_window.geometry("600x400")

    # Thanh tìm kiếm ảo
    entry = Entry(keyboard_window, font=("Arial", 24), width=30)
    entry.pack(pady=20)

    # Các nút bàn phím
    keys = [
        'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P',
        'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L',
        'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'Space', 'Backspace'
    ]

    # Hiển thị các nút trên bàn phím
    row = 3
    col = 0
    for key in keys:
        btn = tk.Button(keyboard_window, text=key, font=("Arial", 18), width=5, height=2,
                        command=lambda k=key: on_key_press(k))
        btn.grid(row=row, column=col, padx=5, pady=5)
        col += 1
        if key in {'P', 'L'}:  # Xuống dòng sau các phím này
            row += 1
            col = 0

    keyboard_window.mainloop()

# Nhận diện gương mặt qua webcam
def face_recognition_mode():
    # Mở webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Chuyển khung hình sang định dạng RGB
        rgb_frame = frame[:, :, ::-1]

        # Tìm các gương mặt trong khung hình
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            # So sánh gương mặt hiện tại với gương mặt đã biết
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                matched_index = matches.index(True)
                name = known_face_names[matched_index]
                print(f"Gương mặt đã nhận diện: {name}")

                # Đóng webcam và chuyển sang chế độ bàn phím ảo
                video_capture.release()
                cv2.destroyAllWindows()
                show_virtual_keyboard()
                return

        # Hiển thị webcam
        cv2.imshow('Video', frame)

        # Thoát chương trình khi nhấn "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Đóng webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Tải gương mặt đã biết
    load_known_faces()
    # Bắt đầu chế độ nhận diện gương mặt
    face_recognition_mode()