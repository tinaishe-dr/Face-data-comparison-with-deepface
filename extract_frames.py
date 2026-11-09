
from pathlib import Path
import cv2

VIDEO_DIR = Path("dataset/selfie_videos")
OUTPUT_DIR = Path("dataset/selfie_frames")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for video_path in sorted(VIDEO_DIR.glob("*.mp4")):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Invalid FPS: {video_path.name}")
        cap.release()
        continue

    frame_interval = max(1, round(fps))
    frame_index = 0
    saved = 0

    subject_dir = OUTPUT_DIR / video_path.stem
    subject_dir.mkdir(parents=True, exist_ok=True)

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % frame_interval == 0:
            timestamp = frame_index / fps
            output_path = (
                subject_dir /
                f"frame_{timestamp:.2f}s.jpg"
            )

            if cv2.imwrite(str(output_path), frame):
                saved += 1

        frame_index += 1

    cap.release()

    print(f"{video_path.name}: saved {saved} frames")

print("Frame extraction completed.")
