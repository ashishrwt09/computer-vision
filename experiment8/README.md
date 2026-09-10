# Experiment 8: Real-Time Object Tracking & Motion Analysis

## Questions & Answers

### Q1. How does object tracking differ from object detection?
**Answer:**
- **Object Detection:** Identifies *what* an object is and *where* it is in a single frame (e.g., bounding box for a car). It processes each frame independently.
- **Object Tracking:** Follows the movement of a detected object *across multiple continuous frames*, calculating its trajectory and velocity.

---

### Q2. Explain how Optical Flow can be used for real-time object tracking.
**Answer:**
Optical Flow tracks motion vectors between consecutive frames. By locating key feature points on an object in Frame 1, optical flow predicts where those exact points move in Frame 2, enabling real-time trajectory tracking without needing to re-detect the object from scratch.

---

### Q3. What is the role of Shi-Tomasi Corner Detection in the Lucas-Kanade Optical Flow algorithm?
**Answer:**
Lucas-Kanade tracks **sparse keypoints**. Shi-Tomasi Corner Detection finds high-contrast, distinct points (like corners and edges) in the first frame that are easy to track accurately across video frames.

---

### Q4. Why is Optical Flow suitable for motion analysis in videos?
**Answer:**
- **Speed:** Faster than running heavy object detectors on every frame.
- **Continuous Velocity Data:** Directly provides motion magnitude (speed) and direction (angle).
- **No Training Needed:** Works on any moving object without requiring pre-trained AI models.

---

### Q5. What challenges arise while tracking fast-moving or partially occluded objects?
**Answer:**
- **Fast Motion:** Violates the small-motion assumption, causing feature points to be lost.
- **Occlusion:** When an object passes behind another, its feature points disappear, breaking the tracking path.

---

### Q6. Compare Optical Flow-based tracking with deep learning-based object tracking methods.
**Answer:**

| Feature | Optical Flow Tracking | Deep Learning Tracking (e.g., YOLO + SORT) |
| :--- | :--- | :--- |
| **Computation** | Fast, lightweight (runs on CPU) | Heavy (requires GPU) |
| **Object Class** | Tracks any pixel motion | Identifies specific trained classes |
| **Occlusion Handling** | Weak | Stronger (re-identifies object) |

---

### Q7. How can Optical Flow be used in traffic monitoring and autonomous driving systems?
**Answer:**
- **Traffic Monitoring:** Measures vehicle counts, estimates average road speed, and flags wrong-way drivers.
- **Autonomous Driving:** Predicts pedestrian/vehicle movement trajectories and estimates Time-to-Collision (TTC).

---

### Q8. Explain the effect of camera motion on Optical Flow estimation.
**Answer:**
When the camera moves (panning/zooming), **all pixels move in the background**. This creates global background optical flow, making it harder to separate the actual moving object from background motion without camera ego-motion compensation.

---

### Q9. Mention five real-world applications where motion analysis using Optical Flow is commonly employed.
**Answer:**
1. **Autonomous Vehicles:** Collision avoidance and pedestrian tracking.
2. **Video Surveillance:** Automated intrusion detection.
3. **Sports Analytics:** Player speed and ball trajectory tracking.
4. **Robot Navigation:** Visual Odometry and obstacle avoidance.
5. **Human Activity Recognition:** Gesture control and posture analysis.

---

### Q10. How can Optical Flow improve the performance of surveillance, robotics, and human activity recognition systems?
**Answer:**
- **Surveillance:** Quickly highlights moving areas without scanning static backgrounds.
- **Robotics:** Helps robots calculate self-motion and navigate safely.
- **Human Activity:** Captures motion direction/patterns (like running vs. falling) for action recognition models.
