import cv2
from driver_monitoring.face_eye_tracking import FaceEyeTracker

cap = cv2.VideoCapture(0)
tracker = FaceEyeTracker(resize_factor=0.5)  # Resize for speed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ear, alert = tracker.process_frame(frame)
    cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if alert:
        cv2.putText(frame, "DROWSINESS ALERT!", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Advanced Face & Eye Tracker", frame)

    # Stop button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tracker.close()
cap.release()
cv2.destroyAllWindows()
