import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Same folder (experiment2) se image uthayega
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(CURRENT_DIR, 'IMG_20260101_085451.jpg.jpeg')

# Read image in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Image nahi mili: {image_path}")

# 2. Contrast Stretching (Min-Max Normalization to [0, 255])
r_min, r_max = int(np.min(img)), int(np.max(img))
if r_max > r_min:
    img_stretched = ((img.astype(np.float32) - r_min) / (r_max - r_min) * 255.0).astype(np.uint8)
else:
    img_stretched = img.copy()

# 3. Global Histogram Equalization (HE)
img_he = cv2.equalizeHist(img)

# 4. Contrast Limited Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img)

# 5. Plot Comparison and Histograms
images = [img, img_stretched, img_he, img_clahe]
titles = ['Original (Low Contrast)', 'Contrast Stretched', 'Global HE', 'CLAHE']

plt.figure(figsize=(16, 8))
for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[i], fontsize=11)
    plt.axis('off')

    plt.subplot(2, 4, i + 5)
    plt.hist(images[i].ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    plt.title(f'Histogram: {titles[i]}', fontsize=10)
    plt.xlim([0, 256])
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')

plt.tight_layout()

# Save result in experiment2 folder
output_path = os.path.join(CURRENT_DIR, 'contrast_enhancement_result.png')
plt.savefig(output_path)
plt.show()