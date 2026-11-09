
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("dataset")
RESULTS = Path("results")

RESULTS.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "DSLR": ROOT / "dslr_images",
    "Selfie": ROOT / "selfie_frames",
}

DETECTOR = "opencv"
FACE_CROP_SIZE = (224, 224)

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}


# ============================================================
# IMAGE QUALITY FUNCTIONS
# ============================================================

def calculate_sharpness(image):
    """
    Calculate Laplacian variance.

    Higher values generally indicate more image detail,
    but this is not a calibrated sharpness score.
    """
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    return float(
        cv2.Laplacian(
            gray, cv2.CV_64F
        ).var()
    )


def calculate_brightness(image):
    """
    Calculate mean grayscale brightness (0-255).
    """
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    return float(np.mean(gray))


def calculate_contrast(image):
    """
    Calculate grayscale intensity standard deviation.
    """
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    return float(np.std(gray))


def crop_face(image, facial_area):
    """
    Crop the detected face from the original image.
    Clamp coordinates to valid image boundaries.
    """
    height, width = image.shape[:2]

    x = int(facial_area["x"])
    y = int(facial_area["y"])
    w = int(facial_area["w"])
    h = int(facial_area["h"])

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def calculate_face_sharpness(image, facial_area):
    """
    Calculate sharpness on a standardized face crop.

    Resizing makes the measurement more comparable
    across images of different resolutions.
    """
    face = crop_face(image, facial_area)

    if face is None:
        return None

    resized_face = cv2.resize(
        face,
        FACE_CROP_SIZE,
        interpolation=cv2.INTER_AREA
    )

    return calculate_sharpness(resized_face)


def calculate_face_position(facial_area, image_width,
                            image_height):
    """
    Return normalized face center coordinates.

    0.5, 0.5 means the bounding box is centered.
    """
    x = facial_area["x"]
    y = facial_area["y"]
    w = facial_area["w"]
    h = facial_area["h"]

    center_x = (x + w / 2) / image_width
    center_y = (y + h / 2) / image_height

    return center_x, center_y


def calculate_face_area_ratio(facial_area,
                              image_width,
                              image_height):
    """
    Proportion of image covered by face bounding box.
    """
    face_area = (
        facial_area["w"] * facial_area["h"]
    )

    image_area = image_width * image_height

    if image_area == 0:
        return None

    return face_area / image_area


def calculate_face_dimensions(facial_area,
                              image_width,
                              image_height):
    """
    Calculate normalized face width and height.
    """
    face_width_ratio = (
        facial_area["w"] / image_width
    )

    face_height_ratio = (
        facial_area["h"] / image_height
    )

    return face_width_ratio, face_height_ratio


def get_eye_landmarks(facial_area):
    """
    Check whether the detector returned both eye landmarks.

    This does not prove that the eyes are unobstructed
    or clearly visible.
    """
    left_eye = facial_area.get("left_eye")
    right_eye = facial_area.get("right_eye")

    return (
        left_eye is not None,
        right_eye is not None
    )


# ============================================================
# REVIEW SINGLE IMAGE
# ============================================================

def review_image(image_path, source):
    """
    Review one image and return its measurements.
    """

    start_time = time.perf_counter()

    row = {
        "source": source,
        "image": str(image_path),
        "filename": image_path.name,

        "width": None,
        "height": None,

        "full_image_sharpness": None,
        "face_sharpness": None,

        "full_image_brightness": None,
        "face_brightness": None,

        "full_image_contrast": None,
        "face_contrast": None,

        "faces_detected": 0,
        "detection_confidence": None,

        "face_x": None,
        "face_y": None,
        "face_width": None,
        "face_height": None,

        "face_area_ratio": None,
        "face_width_ratio": None,
        "face_height_ratio": None,

        "face_center_x": None,
        "face_center_y": None,

        "left_eye_detected": False,
        "right_eye_detected": False,

        "status": "NOT_PROCESSED",
        "error": "",
        "processing_time_seconds": None,
    }

    try:
        image = cv2.imread(str(image_path))

        if image is None:
            row["status"] = "UNREADABLE_IMAGE"
            row["error"] = "OpenCV could not read image"
            return row

        height, width = image.shape[:2]

        row["width"] = width
        row["height"] = height

        # Full-image quality
        row["full_image_sharpness"] = (
            calculate_sharpness(image)
        )

        row["full_image_brightness"] = (
            calculate_brightness(image)
        )

        row["full_image_contrast"] = (
            calculate_contrast(image)
        )

        # Face detection
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend=DETECTOR,
            enforce_detection=True,
            align=True,
        )

        row["faces_detected"] = len(faces)

        if len(faces) == 0:
            row["status"] = "NO_FACE"
            return row

        if len(faces) > 1:
            row["status"] = "MULTIPLE_FACES"
            return row

        # Single face
        face = faces[0]
        area = face["facial_area"]

        row["detection_confidence"] = (
            face.get("confidence")
        )

        row["face_x"] = area["x"]
        row["face_y"] = area["y"]
        row["face_width"] = area["w"]
        row["face_height"] = area["h"]

        row["face_area_ratio"] = (
            calculate_face_area_ratio(
                area, width, height
            )
        )

        fw_ratio, fh_ratio = (
            calculate_face_dimensions(
                area, width, height
            )
        )

        row["face_width_ratio"] = fw_ratio
        row["face_height_ratio"] = fh_ratio

        center_x, center_y = (
            calculate_face_position(
                area, width, height
            )
        )

        row["face_center_x"] = center_x
        row["face_center_y"] = center_y

        left_eye, right_eye = (
            get_eye_landmarks(area)
        )

        row["left_eye_detected"] = left_eye
        row["right_eye_detected"] = right_eye

        # Face-only quality
        face_crop = crop_face(image, area)

        if face_crop is None:
            row["status"] = "INVALID_FACE_CROP"
            return row

        row["face_sharpness"] = (
            calculate_face_sharpness(image, area)
        )

        row["face_brightness"] = (
            calculate_brightness(face_crop)
        )

        row["face_contrast"] = (
            calculate_contrast(face_crop)
        )

        row["status"] = "REVIEWED"

    except Exception as error:
        row["status"] = "ERROR"
        row["error"] = str(error)

    finally:
        row["processing_time_seconds"] = round(
            time.perf_counter() - start_time, 4
        )

    return row


# ============================================================
# PROCESS DATASET
# ============================================================

def main():

    print("=" * 60)
    print("FACE DATASET QUALITY REVIEW")
    print("=" * 60)

    rows = []

    for source, folder in SOURCES.items():

        print(f"\nProcessing {source} images...")

        if not folder.exists():
            print(f"Folder not found: {folder}")
            continue

        image_paths = sorted(
            path for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        print(f"Images found: {len(image_paths)}")

        for index, image_path in enumerate(
            image_paths, start=1
        ):

            print(
                f"[{index}/{len(image_paths)}] "
                f"{image_path.name}"
            )

            result = review_image(
                image_path, source
            )

            rows.append(result)

            print(
                f"  Status: {result['status']} | "
                f"Face sharpness: "
                f"{result['face_sharpness']}"
            )

    if not rows:
        print("\nNo images were found.")
        return

    # Export detailed report
    df = pd.DataFrame(rows)

    output_path = RESULTS / "face_review.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("REVIEW SUMMARY")
    print("=" * 60)

    print("\nImages by source:")
    print(df.groupby("source").size())

    print("\nProcessing status:")
    print(
        pd.crosstab(
            df["source"],
            df["status"]
        )
    )

    reviewed = df[
        df["status"] == "REVIEWED"
    ]

    if not reviewed.empty:

        print("\nAverage face sharpness:")
        print(
            reviewed.groupby("source")[
                "face_sharpness"
            ].mean()
        )

        print("\nAverage face brightness:")
        print(
            reviewed.groupby("source")[
                "face_brightness"
            ].mean()
        )

        print("\nAverage face area ratio:")
        print(
            reviewed.groupby("source")[
                "face_area_ratio"
            ].mean()
        )

    print(f"\nReport saved to: {output_path}")
    print("Review completed.")


if __name__ == "__main__":
    main()
