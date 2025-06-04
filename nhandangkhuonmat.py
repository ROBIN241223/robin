import cv2
import face_recognition
import numpy as np
import os


def load_known_faces(directory):
    """Tải các khuôn mặt và tên từ thư mục."""
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            # Xử lý trường hợp có nhiều khuôn mặt hoặc không có khuôn mặt
            if len(face_encodings) > 0:
                for face_encoding in face_encodings:  # Lặp qua tất cả các khuôn mặt
                    known_face_encodings.append(face_encoding)
                    # Lấy tên từ tên file, xử lý trường hợp nhiều người trong ảnh
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name)
            else:
                print(f"Cảnh báo: Không tìm thấy khuôn mặt nào trong {filename}")

    return known_face_encodings, known_face_names


def recognize_faces(known_face_encodings, known_face_names):
    """Nhận dạng khuôn mặt từ webcam."""
    video_capture = cv2.VideoCapture(0)  # Sử dụng webcam mặc định (0)

    if not video_capture.isOpened():
        print("Không thể mở webcam.")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Lỗi khi đọc frame từ webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #rgb_frame = frame[:, :, ::-1] # Một cách khác, tương đương

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # Kiểm tra xem face_distances có rỗng không (quan trọng!)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # Vẽ hộp và tên
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    known_faces_dir = "known_faces"

    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        print(f"Đã tạo thư mục '{known_faces_dir}'. Thêm ảnh khuôn mặt vào đó (tên file là tên người).")
        exit()

    known_encodings, known_names = load_known_faces(known_faces_dir)

    if not known_encodings:
        print("Không tìm thấy khuôn mặt hợp lệ nào trong thư mục 'known_faces'.")
    else:
        recognize_faces(known_encodings, known_names)