# Experiment 7: Optical Flow Q&A

### Q1. What is Optical Flow? How is it used in computer vision?
**Answer:**
Optical Flow calculates how objects or pixels move between two consecutive video frames. 

**Uses:**
- Tracking moving objects (e.g., cars, people).
- Camera motion estimation.
- Gesture and human action recognition.

---

### Q2. Explain the working principle of the Lucas-Kanade Optical Flow algorithm.
**Answer:**
Lucas-Kanade is a **sparse** optical flow algorithm. It tracks only specific feature points (like corners) instead of every pixel.

It assumes that:
1. Pixel brightness doesn't change much from one frame to the next.
2. Nearby pixels move in the same direction together.

---

### Q3. What is Dense Optical Flow? How does it differ from Sparse Optical Flow?
**Answer:**
- **Sparse Optical Flow:** Calculates motion for only selected key points (fast & lightweight).
- **Dense Optical Flow:** Calculates motion for **every single pixel** in the video frame (detailed & accurate, but slower).

---

### Q4. Compare the Lucas-Kanade and Farneback optical flow algorithms.
**Answer:**

| Parameter | Lucas-Kanade | Farneback |
| :--- | :--- | :--- |
| **Type** | Sparse (tracks key points) | Dense (tracks all pixels) |
| **Speed** | Fast | Slower |
| **Best For** | Real-time object tracking | Scene analysis & full motion fields |

---

### Q5. What assumptions are made while computing optical flow?
**Answer:**
1. **Brightness Constancy:** Object brightness stays the same across frames.
2. **Small Motion:** Movement between frames is small.
3. **Spatial Coherence:** Neighboring pixels move together.

---

### Q6. What factors can affect the accuracy of optical flow estimation?
**Answer:**
- Sudden changes in lighting or shadows.
- Fast-moving objects.
- Plain/textureless surfaces (like a blank wall).
- Camera blur or shake.

---

### Q7. Mention five real-world applications of optical flow in computer vision.
**Answer:**
1. **Self-Driving Cars:** Obstacle and pedestrian detection.
2. **Surveillance:** Identifying suspicious motion or intrusions.
3. **Sports Analytics:** Tracking players and ball speed.
4. **Video Stabilization:** Removing camera shake from videos.
5. **Gesture Control:** Controlling devices with hand movements.

---

### Q8. Why are grayscale images generally used for optical flow computation?
**Answer:**
- **Faster:** Grayscale has 1 color channel instead of 3 (RGB), making calculations 3x faster.
- **Simpler:** Motion estimation relies on brightness patterns, not color.

---

### Q9. What are the limitations of optical flow algorithms in real-world environments?
**Answer:**
- Fails if lights flicker or shadows change suddenly.
- Requires high computing power for dense flow on high-res video.
- Difficulty in tracking motion on flat, smooth surfaces without edges.

---

### Q10. How does optical flow contribute to autonomous driving, surveillance, and action recognition?
**Answer:**
- **Autonomous Driving:** Detects moving vehicles/pedestrians and predicts collisions.
- **Surveillance:** Detects unusual movement in static background scenes.
- **Action Recognition:** Identifies human actions (like walking, running, or jumping) by analyzing movement direction and speed.
