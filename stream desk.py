import serial
import keyboard
import os
import time
import webbrowser
import pyautogui

try:
    ser = serial.Serial('COM9', 9600)  # Thay COM9 bằng cổng COM của Arduino
except serial.SerialException as e:
    print(f"Lỗi khi mở cổng Serial: {e}")
    time.sleep(2)
    try:
        ser = serial.Serial('COM9', 9600)
    except serial.SerialException as e:
        print(f"Lỗi khi mở cổng Serial lần 2: {e}")
        exit()

while True:
    try:
        line = ser.readline().decode('utf-8').rstrip()
        #if line == 'A':
        if line == 'button1':
            os.system('start microsoft.windows.camera:')  # Mở camera
        #elif line == 'B':
        elif line == 'button2':
            #keyboard.press('win')
            #keyboard.press('shift')
            #keyboard.press('s')
            #keyboard.release('s')
            #keyboard.release('shift')
            #keyboard.release('win')
            #print("Đã bật chế độ chụp màn hình")

            #webbrowser.open("https://www.youtube.com/")  # Mở youtube
            #print("Đã mở YouTube")

            pyautogui.press('F')  # giả lập bấm phím space.
            #print("Đã phóng to màn hình")

            #pyautogui.copy(copy) #copy

            #timestamp = time.strftime("%Y%m%d-%H%M%S") #tạo tên file theo ngày giờ
            #filename = f"screenshot_{timestamp}.png"
            #pyautogui.screenshot(filename) #chụp màn hình và lưu file
            #print(f"Đã chụp ảnh màn hình và lưu thành {filename}")
        elif line == 'button3':
            #pyautogui.press('numadd')  # Giả lập bấm phím Num+
            #print("Đã bấm phím Num+")
            webbrowser.open("https://www.youtube.com/")  # Mở youtube
            #print("Đã mở YouTube")
        elif line == 'button4':
            webbrowser.open("https://www.facebook.com/")  # Mở facebook
            #print("Đã mở FACEBOOK")

            #os.system(r'start C:\Program Files\TikTok LIVE Studio\TikTok LIVE Studio Launcher.exe')  # Mở camera
        elif line == 'button5':
            webbrowser.open("https://www.facebook.com/messages/e2ee/t/8204229809611709")  # Mở facebook
            #print("Đã mở messenger")


    except UnicodeDecodeError as e:
        print(f"Lỗi giải mã dữ liệu Serial: {e}")
    except serial.SerialException as e:
        print(f"Lỗi cổng Serial: {e}")
        break

ser.close()