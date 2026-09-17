import os
import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Output directory path set karein
output_dir = 'experiment9'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Pre-trained YOLOv8 Nano model load karein
model = YOLO('yolov8n.pt')

# Step 2: Sample test image download karke experiment9 folder me save karein
image_url = 'https://ultralytics.com/images/bus.jpg'
image_bytes = requests.get(image_url).content

image_path = os.path.join(output_dir, 'sample_image.jpg')
with open(image_path, 'wb') as f:
    f.write(image_bytes)

# Step 3: Inference / Object Detection run karein
results = model(image_path)

# Step 4: Bounding boxes aur labels process karein
for result in results:
    annotated_frame = result.plot()
    
    print("\n--- Detection Results ---")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = result.names[cls_id]
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
        print(f"Object: {label} | Confidence: {confidence:.2f} | BBox: {[round(x, 1) for x in bbox]}")

# Step 5: Final output image ko experiment9 folder ke andar save karein
output_path = os.path.join(output_dir, 'output_detection.png')

output_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 6))
plt.imshow(output_rgb)
plt.axis('off')
plt.title('YOLOv8 Object Detection Result')
plt.savefig(output_path, bbox_inches='tight')
plt.close()

print(f"\nOutput saved successfully inside: {output_path}")