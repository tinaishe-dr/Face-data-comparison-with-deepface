
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/face_review.csv")

df.boxplot(column="blur_score", by="source")
plt.title("Image sharpness by capture source")
plt.suptitle("")
plt.ylabel("Laplacian variance")
plt.tight_layout()
plt.show()
