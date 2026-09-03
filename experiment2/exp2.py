import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Load image (Replace or use your local/workspace image)
image_path = 'IMG_20260101_085451.jpg.jpeg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    # Fallback dummy low-contrast image if path does not exist
    img = np.random.randint(80, 140, (400, 400), dtype=np.uint8)

# 2. Contrast Stretching (Min-Max Normalization to [0, 255])
r_min, r_max = np.min(img), np.max(img)
if r_max > r_min:
    img_stretched = ((img - r_min) / (r_max - r_min) * 255.0).astype(np.uint8)
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
    # Display Processed Images
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[i], fontsize=11)
    plt.axis('off')

    # Display Corresponding Histograms
    plt.subplot(2, 4, i + 5)
    plt.hist(images[i].ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    plt.title(f'Histogram: {titles[i]}', fontsize=10)
    plt.xlim([0, 256])
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')

plt.tight_layout()
plt.savefig('experiment2/contrast_enhancement_result.png')
plt.show()