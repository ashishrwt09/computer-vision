#part-a

import cv2
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

# Online Image load kar rahe hain taaki path ka koi issue na aaye
url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

# Conversions
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 3: Gaussian Blur
gaussian_blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)

# Step 4: Median Filter
median_filter = cv2.medianBlur(img_rgb, 5)

# Step 5: Average Filter
average_filter = cv2.blur(img_rgb, (5, 5))

# Step 6: Laplacian Filter
laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)

# Step 7: Sobel Edge Detection
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_abs = cv2.convertScaleAbs(sobel_combined)

# Plotting and Saving Output Image
titles = [
    "Original Image",
    "Gaussian Blur",
    "Median Filter",
    "Average Filter",
    "Laplacian Filter",
    "Sobel Edge Detection",
]
images = [
    img_rgb,
    gaussian_blur,
    median_filter,
    average_filter,
    laplacian_abs,
    sobel_abs,
]

plt.figure(figsize=(15, 10))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap="gray")
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.savefig("output.png")
print(
    "Execution complete! Output image saved as 'output.png' in your workspace."
)



#part-b

# 1. What is spatial filtering? How is it used in digital image processing?
# Sol:
# Spatial filtering is a technique where pixel values are recalculated directly using the values of neighboring pixels via a spatial mask (kernel). In digital image processing, it is used for noise reduction, image smoothing, edge detection, and detail enhancement.

# 2. Differentiate between Low-Pass Filters and High-Pass Filters with suitable examples.
# sol:
# Feature	Low-Pass Filter (LPF)	High-Pass Filter (HPF)
# Passes	Low spatial frequencies (smooth areas)	High spatial frequencies (edges, fine details)
# Attenuates	Rapid intensity changes (sharp transitions)	Constant intensity or slow transitions
# Primary Effect	Image smoothing, blurring, and noise reduction	Edge enhancement and detail sharpening
# Examples	Gaussian Blur, Box/Average Filter, Median Filter	Sobel Operator, Laplacian Filter, Prewitt Filter

# 3. Compare Average Filter, Gaussian Filter, and Median Filter based on their working principles and applications.
# sol:
# Filter	Working Principle	Best Applications
# Average (Mean)	Replaces center pixel with the arithmetic mean of all pixels in the neighborhood window.	General background smoothing; fast computational reduction of random noise.
# Gaussian	Replaces center pixel with a weighted average using a 2D Gaussian bell-curve distribution (nearer pixels get higher weight).	Preserving structural edges better than average blur while removing Gaussian noise.
# Median	Replaces center pixel with the median value of sorted neighboring pixels (non-linear filter).	Removing salt-and-pepper (impulse) noise without blurring sharp edges.

# 4. Why is the Median Filter particularly effective for removing salt-and-pepper noise?
# sol:
# Salt-and-pepper noise introduces extreme high and low intensity values (0 or 255) as isolated pixels. Because the median filter sorts pixel values in a neighborhood window and picks the middle rank, these extreme values are pushed to the boundaries of the sorted array and discarded, avoiding averaging noisy values into clean regions.

# 5.Explain the role of convolution kernels in spatial filtering.
# sol:
# A convolution kernel is a small matrix (e.g., 3×3 or 5×5) containing spatial weights. As it slides across the image pixel-by-pixel, a element-wise multiplication and summation (convolution) is calculated to generate the target pixel value. The weights inside the kernel determine the filter behavior—equal positive weights smooth images, whereas contrasting positive and negative weights accentuate spatial gradients and edges.

# 6. What is the purpose of the Sobel and Laplacian operators in edge detection?
# sol:
# Sobel Operator: Uses first-order derivatives (X and Y directional gradients) to calculate gradient magnitude and direction, identifying major directional edges while suppressing noise.

# Laplacian Operator: Uses a second-order derivative to find regions of rapid intensity change (zero-crossings), producing fine isotropic (direction-independent) edge boundaries.

# 7. Why are filtering operations considered an essential preprocessing step in computer vision?
# sol:
# Filtering cleans raw image data by eliminating sensor noise, lighting variations, and artifact interference. By enhancing key visual structures (edges, textures) and suppressing irrelevant variations, spatial filtering improves the accuracy and stability of downstream computer vision tasks like segmentation, object recognition, and feature tracking.

# 8. Discuss the trade-off between image smoothing and edge preservation during filtering.
# sol:
# Image smoothing reduces high-frequency noise by averaging pixel values across spatial neighborhoods. However, edges are also high-frequency structures; aggressive linear smoothing (like mean or Gaussian filtering) inadvertently blurs sharp boundaries, reducing detail clarity. Non-linear or adaptive filters (like median or bilateral filters) aim to balance this trade-off by preserving spatial discontinuities.

# 9. Mention four real-world applications where spatial filtering techniques are widely used.
# sol:
# Medical Image Processing: Enhancement of MRI and X-ray images for tumor and bone fracture visualization.
# Autonomous Driving: Edge detection and gradient filtering for lane tracking and obstacle detection.\
# Satellite Remote Sensing: Smoothing noise artifacts and sharpening terrain edges in satellite imagery.
# Document Processing & OCR: Noise removal and detail enhancement for text extraction from scanned documents.

# 10. Compare spatial domain filtering with frequency domain filtering in terms of implementation and practical applications.
# sol:
# Spatial Domain Filtering
# Implementation: Direct pixel values par small matrix kernels (3×3, 5×5) slide karke 2D spatial convolution execute kiya jata hai.
# Complexity: Mathematical calculation simple hoti hai aur chote kernels ke liye ye kafi fast aur lightweight kaam karta hai.
# Practical Applications: Real-time edge detection (Sobel, Laplacian), local noise reduction (Median, Gaussian blur), aur basic image enhancement me use hota hai.

# Frequency Domain Filtering
# Implementation: Image ko pehle Fast Fourier Transform (FFT) se frequency domain me transform karte hain, filter mask multiply karte hain, aur phir Inverse FFT (IFFT) se result wapas late hain.
# Complexity: Transform calculate karne ka extra computational overhead hota hai, par large global filters ke liye ye spatial convolution se zyada efficient padta hai.
# Practical Applications: Global periodic noise remove karne, sharp frequency cut-off design karne, aur image me specific spatial structural patterns isolate karne me use hota hai.