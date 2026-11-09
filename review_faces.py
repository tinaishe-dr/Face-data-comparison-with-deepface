
from pathlib import Path
import cv2
import pandas as pd
from deepface import DeepFace

ROOT = Path("dataset")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

rows = []

sources = {
    "DSLR": ROOT / "dslr_images",
    "Selfie": ROOT / "selfie_frames",
}

for source, folder in sources.items():
    if not folder.exists():
        continue

    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        image = cv2.imread(str(path))

        if image is None:
            print(f"Unreadable: {path}")
            continue

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Laplacian variance: a rough focus/sharpness indicator
        blur_score = cv2.Laplacian(
            gray, cv2.CV_64F
        ).var()

        row = {
            "source": source,
            "image": str(path),
            "width": width,
            "height": height,
            "blur_score": round(float(blur_score), 2),
            "faces_detected": 0,
            "face_area_ratio": None,
            "detection_confidence": None,
            "error": "",
        }

        try:
            faces = DeepFace.extract_faces(
                img_path=str(path),
                detector_backend="opencv",
                enforce_detection=True,
                align=True,
            )

            row["faces_detected"] = len(faces)

            if len(faces) == 1:
                face = faces[0]
                area = face["facial_area"]

                ratio = (
                    area["w"] * area["h"]
                ) / (width * height)

                row["face_area_ratio"] = round(ratio, 4)
                row["detection_confidence"] = face.get(
                    "confidence"
                )

        except Exception as error:
            row["error"] = str(error)

        rows.append(row)
        print(f"Processed: {path.name}")

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "face_review.csv", index=False)

print("\nReview summary:")
if not df.empty:
    print(df.groupby("source")["faces_detected"].agg(
        ["count", "mean"]
    ))

print("\nResults saved to results/face_review.csv")
