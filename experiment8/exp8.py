import os
import cv2
import numpy as np

FOLDER = "experiment8"
os.makedirs(FOLDER, exist_ok=True)


# ---------------------------------------------------------
# Helper: Create Synthetic Test Video
# ---------------------------------------------------------
def create_sample_video(filename=os.path.join(FOLDER, "sample.mp4"), width=640, height=480, fps=30, duration=3):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame_idx in range(fps * duration):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40
        x = int(100 + frame_idx * 6) % width
        y = int(240 + np.sin(frame_idx * 0.15) * 60)
        cv2.circle(frame, (x, y), 25, (0, 255, 0), -1)
        out.write(frame)

    out.release()
    print(f"Sample video generated: {filename}")


# ---------------------------------------------------------
# 1. Lucas-Kanade Object Tracking & Motion Analysis
# ---------------------------------------------------------
def run_lucas_kanade_tracking(video_path):
    cap = cv2.VideoCapture(video_path)

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    ret, old_frame = cap.read()
    if not ret:
        print("Error reading video.")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)

    total_displacement = 0.0
    frame_count = 0
    img = old_frame.copy()

    while cap.isOpened() and frame_count < 60:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()

                    dx, dy = a - c, b - d
                    dist = np.sqrt(dx**2 + dy**2)
                    total_displacement += dist

                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 255), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 4, (0, 0, 255), -1)

                img = cv2.add(frame, mask)
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

        frame_count += 1

    avg_speed = total_displacement / max(frame_count, 1)
    cv2.putText(img, f"Avg Speed: {avg_speed:.2f} px/frame", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = os.path.join(FOLDER, "tracking_lucas_kanade.png")
    cv2.imwrite(out_path, img)
    print(f"Lucas-Kanade Tracking result saved: {out_path}")
    print(f"Estimated Avg Speed: {avg_speed:.2f} px/frame")
    cap.release()


# ---------------------------------------------------------
# 2. Farneback Dense Motion Analysis
# ---------------------------------------------------------
def run_farneback_dense(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    if not ret:
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
        flow = cv2.calcOpticalFlowFarneback(prvs, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        prvs = next_gray
        frame_count += 1

    out_path = os.path.join(FOLDER, "motion_dense_farneback.png")
    cv2.imwrite(out_path, bgr)
    print(f"Farneback Dense Motion result saved: {out_path}")
    cap.release()


if __name__ == "__main__":
    video_file = os.path.join(FOLDER, "sample.mp4")

    if not os.path.exists(video_file):
        create_sample_video(video_file)

    print("Running Lucas-Kanade Motion Tracking...")
    run_lucas_kanade_tracking(video_file)

    print("Running Farneback Dense Motion Field...")
    run_farneback_dense(video_file)

    print("Experiment 8 Execution Complete!")