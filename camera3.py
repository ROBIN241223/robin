import cv2
import time
import os
import hand as htm
import mediapipe


# Khởi tạo các biến
pTime = 0
cap = cv2.VideoCapture(0)

# Đường dẫn đến thư mục chứa hình ảnh
FolderPath = "Fingers"
lst = os.listdir(FolderPath)
lst_2 = []

# Đọc các hình ảnh trong thư mục
for i in lst:
    image = cv2.imread(f"{FolderPath}/{i}")
    if image is not None:  # Kiểm tra xem hình ảnh có được đọc thành công không
        print(f"{FolderPath}/{i}")
        lst_2.append(image)

# Khởi tạo bộ phát hiện tay
detector = htm.handDetector(detectionCon=0)

fingerid = [4,8,12,16,20]

# Vòng lặp chính
while True:
    ret, frame = cap.read()
    if not ret:  # Kiểm tra xem có đọc được khung hình không
        print("Không thể đọc khung hình từ camera.")
        break

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    print(lmList)

    if len(lmList) != 0:
        fingers =[]
        if lmList[fingerid[0]][1] < lmList[fingerid[0]-2][1]:
             fingers.append(1)



        for id in range(1,5):
         print(id)
         if lmList[fingerid[id]][2] < lmList[fingerid[id]-2][2]:
             fingers.append(1)
             print(lmList[fingerid[id]][2])
             print(lmList[fingerid[id]-2][2])
        else:
            fingers.append(0)

        print(fingers)


    # Kiểm tra xem lst_2 có phần tử không



    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    print(type(fps))

    cv2.putText(frame, f"FPS: {int(fps)}", (150, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('ROBIN02', frame)

    if cv2.waitKey(1) == ord("q"):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()