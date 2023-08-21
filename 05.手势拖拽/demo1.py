'''
1. 获取视频流
2. 画个方块
3. 获取关键点左边
4. 判断是否重叠
5. 移动
'''

import time

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green



# 获取视频流
cap = cv2.VideoCapture(1)

# 方块参数
sx, sy, sw = 100, 100, 100

start_time = time.time()
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 1)

        # mediapipe处理
        frame.flags.writeable = False
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


        # 画方块
        cv2.rectangle(img=frame, pt1=(sx, sy), pt2=(sx + sw, sy + sw),
                      color=(0, 255, 0), thickness=-1)


        # 计算帧率
        now = time.time()
        fps = int(1 / (now - start_time))
        start_time = now
        fps_text = "fps:" + str(fps)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, fps_text, (20, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 显示
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
