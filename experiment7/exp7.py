import os
import cv2
import numpy as np


# ---------------------------------------------------------
# Helper: Automatically generate a test video if missing
# ---------------------------------------------------------
def create_sample_video(filename="sample.mp4", width=640, height=480, fps=30, duration=3):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame_idx in range(fps * duration):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        x = int(100 + frame_idx * 5) % width
        y = int(200 + np.sin(frame_idx * 0.1) * 50)
        cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
        out.write(frame)

    out.release()
    print(f"Sample video created: {filename}")


# ---------------------------------------------------------
# 1. Lucas-Kanade Optical Flow (Sparse)
# ---------------------------------------------------------
def run_lucas_kanade(video_path):
    cap = cv2.VideoCapture(video_path)

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    color = np.random.randint(0, 255, (100, 3))

    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)
    img = old_frame.copy()

    frame_count = 0
    while cap.isOpened() and frame_count < 60:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(
                        mask,
                        (int(a), int(b)),
                        (int(c), int(d)),
                        color[i % 100].tolist(),
                        2,
                    )
                    frame = cv2.circle(
                        frame, (int(a), int(b)), 5, color[i % 100].tolist(), -1
                    )

                img = cv2.add(frame, mask)
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

        frame_count += 1

    cv2.imwrite("output_lucas_kanade.png", img)
    print("Lucas-Kanade output saved: output_lucas_kanade.png")
    cap.release()


# ---------------------------------------------------------
# 2. Farneback Optical Flow (Dense)
# ---------------------------------------------------------
def run_farneback(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        return

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    bgr = frame1.copy()

    frame_count = 0
    while cap.isOpened() and frame_count < 60:
        ret, frame2 = cap.read()
        if not ret:
            break

        next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prvs, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        prvs = next_gray
        frame_count += 1

    cv2.imwrite("output_farneback.png", bgr)
    print("Farneback output saved: output_farneback.png")
    cap.release()


if __name__ == "__main__":
    video_file = "sample.mp4"

    if not os.path.exists(video_file):
        create_sample_video(video_file)

    print("Running Lucas-Kanade Optical Flow...")
    run_lucas_kanade(video_file)

    print("Running Farneback Optical Flow...")
    run_farneback(video_file)

    print("Processing Complete!")