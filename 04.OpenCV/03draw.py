import time
import cv2

cap = cv2.VideoCapture(1)
start_time = time.time()

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 1)
        # 计算帧率
        now = time.time()
        fps = int(1 / (now - start_time))
        start_time = now

        fps_text = "fps:" + str(fps)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, fps_text, (20, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
