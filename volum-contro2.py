import cv2
import time
import math
import numpy as np
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        return lmList


# --- Main Program ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

pTime = 0

detector = handDetector(detectionCon=0.7, maxHands=1)

# --- Audio Control Setup ---
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
volBar = 400
vol_tyle = 0

# --- Calibration ---
max_distance = 0  # Initialize max_distance


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    if len(lmList) != 0:
        # Get coordinates of thumb (point 4) and ring finger (point 16)
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
        x2, y2 = lmList[16][1], lmList[16][2] # Ring finger tip

        # Calculate the distance
        distance = math.hypot(x2 - x1, y2 - y1)

        # Draw circles and line for visualization
        cv2.circle(frame, (x1, y1), 15, (255, 0, 255), -1)
        cv2.circle(frame, (x2, y2), 15, (255, 0, 255), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # Center
        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), -1)

        # --- Calibration and Volume Control ---
        # 1.  Find max_distance (when thumb and ring finger are fully extended)

        # We assume the user will show the maximum distance first for calibration.
        if distance > max_distance:
          max_distance = distance
          cv2.putText(frame, "Calibrating...", (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


        # 2.  Map distance to volume *after* calibration
        if max_distance > 0:  # Only control volume *after* max_distance is determined
            # Map the distance to the volume range.  Use 0 as the minimum distance.
            vol = np.interp(distance, [0, max_distance], [minVol, maxVol])
            volBar = np.interp(distance, [0, max_distance], [400, 150])
            vol_tyle = np.interp(distance, [0, max_distance], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)
            cv2.putText(frame, "Volume Control Active", (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            # Visual feedback for minimum distance
            if distance < 20:  # Small threshold for "touching"
                cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)

        # --- Visual Volume Bar ---
        cv2.rectangle(frame, (50, 150), (100, 400), (0, 255, 0), 3)
        cv2.rectangle(frame, (50, int(volBar)), (100, 400), (0, 255, 0), -1)
        cv2.putText(frame, f"{int(vol_tyle)} %", (50, 130), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # --- FPS Calculation and Display ---
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Ga Lai Lap Trinh", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()