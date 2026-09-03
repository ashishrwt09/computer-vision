import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Image using direct path
image_path = r"C:\Users\rawat\Downloads\IMG_20260101_085451.jpg"
img_bgr = cv2.imread(image_path)

if img_bgr is None:
    raise FileNotFoundError(f"Image nahi mili! Path check karo: {image_path}")

# Matplotlib ke liye BGR se RGB convert
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Image Properties
height, width, channels = img_bgr.shape
print("=" * 40)
print("IMAGE PROPERTIES")
print("=" * 40)
print(f"Dimensions (W x H) : {width} x {height}")
print(f"Number of Channels : {channels}")
print(f"Data Type          : {img_bgr.dtype}")
print(f"Total Pixels       : {img_bgr.size // channels}")
print("=" * 40)

# 3. Save image in different formats
cv2.imwrite(r"C:\Users\rawat\Downloads\output_compressed.jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
cv2.imwrite(r"C:\Users\rawat\Downloads\output_lossless.png", img_bgr)
print("Images successfully saved as JPEG and PNG in Downloads folder.")

# 4. Color Space Conversions
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

# 5. Geometric Transformations
# Resizing
img_resized = cv2.resize(img_rgb, (300, 300), interpolation=cv2.INTER_AREA)

# Flipping
img_h_flip = cv2.flip(img_rgb, 1)   # Horizontal
img_v_flip = cv2.flip(img_rgb, 0)   # Vertical

# Rotation (-90 degrees clockwise around center)
center = (width // 2, height // 2)
rot_matrix = cv2.getRotationMatrix2D(center, -90, 1.0)
img_rotated = cv2.warpAffine(img_rgb, rot_matrix, (width, height))

# 6. Image Complement (Negative)
img_negative = 255 - img_rgb

# 7. Region of Interest (ROI) Cropping (Center area)
ymin, ymax = height // 4, 3 * height // 4
xmin, xmax = width // 4, 3 * width // 4
roi = img_rgb[ymin:ymax, xmin:xmax]

# 8. Display Results using Matplotlib
titles = [
    'Original (RGB)', 'Grayscale', 'HSV', 
    'LAB', 'Negative Image', 'Horizontal Flip', 
    'Rotated (-90°)', 'Resized (300x300)', 'Cropped ROI'
]
images = [
    img_rgb, img_gray, img_hsv, 
    img_lab, img_negative, img_h_flip, 
    img_rotated, img_resized, roi
]

plt.figure(figsize=(15, 10))
for i in range(len(images)):
    plt.subplot(3, 3, i + 1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i], fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.show()