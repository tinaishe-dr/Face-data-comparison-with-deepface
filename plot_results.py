
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/face_review.csv")

df = df[df["status"] == "REVIEWED"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df.boxplot(
    column="full_image_sharpness",
    by="source",
    ax=axes[0]
)

axes[0].set_title("Full-image sharpness")
axes[0].set_ylabel("Laplacian variance")

df.boxplot(
    column="face_sharpness",
    by="source",
    ax=axes[1]
)

axes[1].set_title("Face-only sharpness")
axes[1].set_ylabel("Laplacian variance")

plt.suptitle("")
plt.tight_layout()
plt.show()
