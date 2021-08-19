from core.detector import detect
import cv2


cap = cv2.VideoCapture('videoplayback.mp4')
n=0
while cap.isOpened():
    n += 1
    cap.grab()
    if n % 4 == 0:
        success, frame = cap.retrieve()
        if success:
            results=detect(source={'camera_id': ['101'], 'frames':[frame]})
        else:
            break

cap.release()
cv2.destroyAllWindows()

print('Succesfully Executed')

    