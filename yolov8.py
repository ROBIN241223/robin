from ultralytics import YOLO

# Tạo mô hình YOLOv8
model = YOLO('yolov8n.pt')  # Sử dụng mô hình YOLOv8 nhỏ nhất

# Huấn luyện mô hình
model.train(
    data='E:/Sounds/pythonProject/my_app/data/2.jpg',  # File YAML mô tả cấu trúc dữ liệu
    epochs=50,                   # Số vòng lặp huấn luyện
    imgsz=640                    # Kích thước ảnh đầu vào
)