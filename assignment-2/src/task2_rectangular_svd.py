import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def run(data_dir="data", output_dir="figures"):
    image_path = os.path.join(data_dir, "cat_21__1_.png")
    color_image = Image.open(image_path)
    gray_image = color_image.convert("L")

    A = np.array(gray_image, dtype=float)
    m, n = A.shape
    print("Rectangular image matrix shape:", A.shape)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    r = len(S)
    print("Number of singular values available:", r)

    def reconstruct_svd(k):
        Sigma_k = np.zeros((r, r))
        for i in range(k):
            Sigma_k[i, i] = S[i]
        A_k = U @ Sigma_k @ Vt
        return A_k

    os.makedirs(output_dir, exist_ok=True)
    k_values = [50, 100, 160]

    for k in k_values:
        A_k = reconstruct_svd(k)
        error = np.abs(A - A_k)

        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        axes[0].imshow(A, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Original")
        axes[1].imshow(A_k, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"SVD Reconstruction (k={k})")
        axes[2].imshow(error, cmap="gray")
        axes[2].set_title("Error |A - A_k|")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"task2_k{k}.png"), dpi=150)
        plt.close()

        frob_error = np.linalg.norm(A - A_k, ord="fro")
        print(f"k={k}: Frobenius error = {frob_error:.2f}")

    k_range = range(1, r + 1)
    errors = []

    for k in k_range:
        A_k = reconstruct_svd(k)
        errors.append(np.linalg.norm(A - A_k, ord="fro"))

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, errors, color="tab:blue")
    plt.xlabel("Number of retained components (k)")
    plt.ylabel("Frobenius norm error E(k)")
    plt.title("Reconstruction error vs k (Rectangular image, SVD only)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "task2_error_vs_k.png"), dpi=150)
    plt.close()

    np.save(os.path.join(output_dir, "task2_errors_svd.npy"), np.array(errors))

    print(f"Task 2 done. Figures saved in {output_dir}/")


if __name__ == "__main__":
    run()
