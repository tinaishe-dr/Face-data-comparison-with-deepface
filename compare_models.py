
from deepface import DeepFace

image1 = "dataset/test_images/IMG_0586.JPG"
image2 = "dataset/test_images/frame_2.00s.jpg"

models = ["ArcFace", "Facenet512", "SFace"]

for model in models:
    try:
        result = DeepFace.verify(
            img1_path=image1,
            img2_path=image2,
            model_name=model,
            detector_backend="opencv",
            distance_metric="cosine",
            enforce_detection=True,
            align=True,
        )

        print(f"\nModel: {model}")
        print("Verified:", result["verified"])
        print("Distance:", result["distance"])
        print("Threshold:", result["threshold"])

    except Exception as error:
        print(f"{model} failed: {error}")
