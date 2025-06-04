import cv2
from ultralytics import YOLO

model = YOLO("best (2).pt")

list_linhkien = {"cap1": 1, "fet1": 1, "led1": 1, "opto1": 1, "res1": 2, "res2": 1}
list_ = []
cap1 = 0
fet1 = 0
led1 = 0
opto1 = 0
res1 = 0
res2 = 0 
status = False
img = cv2.imread('data\z6478961522824_c51be415a97124b4348b3822344f045c.jpg')
results = model.predict(img, imgsz=640, device="cpu", half=False)
result = results[0]

for box in result.boxes:
    conf = round(box.conf[0].item(), 2)
    if conf > 0.2:
        id = int(box.cls[0].item())
        class_id = result.names[id]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        if class_id == "cap1":
            cap1 += 1
        elif class_id == "fet1":
            fet1 += 1
        elif class_id == "led1":
            led1 += 1
        elif class_id == "opto1":
            opto1 += 1
        elif class_id == "res1":
            res1 += 1
        elif class_id == "res2":
            res2 += 1
        list_.append((class_id))
if list_linhkien["cap1"] == cap1 and list_linhkien["fet1"] == fet1 and list_linhkien["led1"] == led1 and list_linhkien["opto1"] == opto1 and list_linhkien["res1"] == res1 and list_linhkien["res2"] == res2:
    status = True
else: status = False
print(cap1, fet1, led1, opto1, res1, res2)
print(status)
annotated_frame = result.plot()

cv2.imshow('output_image', annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
