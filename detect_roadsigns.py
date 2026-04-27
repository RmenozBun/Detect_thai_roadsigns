import cv2
import numpy as np
import random
from ultralytics import YOLO

# โหลดโมเดล YOLOv8 ที่ฝึกมาแล้ว
model = YOLO('path_to_your_exported_model.pt')  # ใส่ path ของโมเดลที่ฝึกมาแล้ว

# รายการป้ายจราจรที่เป็นโจทย์ (ต้องตรงกับคลาสในไฟล์ .yaml)
traffic_signs = ['Speed Limit 20', 'Speed Limit 30', 'Speed Limit 50', 'Stop', 'Yield']
current_sign = random.choice(traffic_signs)

# ฟังก์ชันแสดงโจทย์ใหม่
def new_task():
    return random.choice(traffic_signs)

# เปิดกล้อง
cap = cv2.VideoCapture(0)

score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # ใช้ YOLO ในการตรวจจับป้ายจราจร
    results = model(frame)

    # ตรวจสอบผลการตรวจจับ
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, class_id = result
        label = traffic_signs[int(class_id)]
        
        # แสดงผลการตรวจจับในเฟรม
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # ตรวจสอบว่าผู้ใช้ทายถูกหรือไม่
        if label == current_sign:
            score += 1
            current_sign = new_task()

    # แสดงโจทย์และคะแนนในเฟรม
    cv2.putText(frame, f'Current Task: {current_sign}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f'Score: {score}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Traffic Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
