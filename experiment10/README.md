# Experiment 10: Character/Digit Classification using CNN

## Aim
To design, train, and evaluate a Convolutional Neural Network (CNN) for digit classification using the standard MNIST dataset and analyze its performance.

---

## Questions & Answers

### Q1. What is image classification? How does it differ from object detection and image segmentation?
- **Image Classification:** Assigns a single categorical label to an entire image without locating spatial boundaries.
- **Object Detection:** Identifies multiple objects, assigns labels, and draws bounding boxes around their locations.
- **Image Segmentation:** Classifies every pixel to outline exact visual boundaries of objects.

---

### Q2. Explain the architecture and working principle of a CNN.
A Convolutional Neural Network (CNN) consists of convolutional layers, pooling layers, activation functions, and fully connected layers. It automatically learns feature hierarchies (edges -> textures -> shapes) by sliding spatial filters (kernels) across input matrices.

---

### Q3. What is the role of Convolutional Layers, Pooling Layers, and Fully Connected Layers in a CNN?
- **Convolutional Layer:** Extracts spatial feature maps using learnable filters.
- **Pooling Layer:** Reduces spatial dimensions (downsampling) to lower computation and prevent overfitting.
- **Fully Connected (Dense) Layer:** Combines extracted features to output final class probability distributions.

---

### Q4. Why is image normalization performed before training a deep learning model?
Normalization scales pixel values (e.g., $0-255$ to $0-1$). It stabilizes gradient descent, speeds up optimization convergence, and prevents exploding/vanishing gradient issues.

---

### Q5. Explain the purpose of activation functions such as ReLU and Softmax in CNNs.
- **ReLU ($\max(0, x)$):** Introduces non-linearity, aiding the learning of complex relationships while solving vanishing gradient problems.
- **Softmax:** Converts final output logits into normalized probability distributions that sum up to 1 across target classes.

---

### Q6. What is a Confusion Matrix? How is it used to evaluate classification performance?
A Confusion Matrix is an $N \times N$ layout comparing Ground Truth classes against Predicted classes. It breaks down classification into True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

---

### Q7. Differentiate between Accuracy, Precision, Recall, and F1-Score with suitable examples.
- **Accuracy:** $\frac{TP + TN}{\text{Total}}$ (Overall correctness ratio).
- **Precision:** $\frac{TP}{TP + FP}$ (Exactness of positive predictions).
- **Recall:** $\frac{TP}{TP + FN}$ (Completeness in detecting actual positives).
- **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ (Harmonic mean balancing Precision and Recall).

---

### Q8. Compare traditional feature-based image classification methods with CNN-based classification.

| Feature | Traditional Methods (e.g., SIFT/HOG + SVM) | CNN-Based Classification |
| :--- | :--- | :--- |
| **Extraction** | Requires handcrafted features | Fully automated feature extraction |
| **Scalability** | Struggles on complex/unstructured data | Scales well with big data and deep networks |
| **Pipeline** | Separate extraction and classification steps | End-to-end learning pipeline |

---

### Q9. Mention five real-world applications of CNN-based image classification.
1. Medical Diagnostics (X-ray/MRI scan disease identification)
2. Biometric Authentication (facial recognition systems)
3. Automated Document Processing (OCR digit/character recognition)
4. Quality Inspection (assembly line defect detection)
5. Autonomous Driving (traffic sign categorization)

---

### Q10. How can techniques such as Data Augmentation, Transfer Learning, and Hyperparameter Tuning improve performance?
- **Data Augmentation:** Increases dataset diversity using random rotations/flips, mitigating overfitting.
- **Transfer Learning:** Fine-tunes pre-trained models on smaller target datasets to boost accuracy.
- **Hyperparameter Tuning:** Optimizes learning rate, batch size, and network depth to improve model stability and accuracy.
