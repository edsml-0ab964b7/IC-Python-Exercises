import cv2

cap = cv2.VideoCapture(1)

# 保存视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID,MJPG

fps = 20
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))
print("fps:", fps, "\nwidth:", w, "\nheight:", h)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 1)
        # write the flipped frame
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
