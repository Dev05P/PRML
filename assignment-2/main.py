import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def run_task1(data_dir="data", output_dir="figures"):
    image_path = os.path.join(data_dir, "cat_21.png")
    color_image = Image.open(image_path)
    gray_image = color_image.convert("L")

    A = np.array(gray_image, dtype=float)
    n = A.shape[0]
    print("Square image matrix shape:", A.shape)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    TOL = 1e-6
    blocks = []
    used = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if used[i]:
            i += 1
            continue
        lam = eigenvalues[i]
        if abs(lam.imag) < TOL:
            blocks.append(("real", i))
            used[i] = True
            i += 1
        else:
            partner = i + 1
            blocks.append(("pair", i, partner))
            used[i] = True
            used[partner] = True
            i += 2

    Q_real = np.zeros((n, n))
    Lambda_real_full = np.zeros((n, n))
    block_start = []
    col = 0
    for block in blocks:
        block_start.append(col)
        if block[0] == "real":
            idx = block[1]
            Q_real[:, col] = eigenvectors[:, idx].real
            Lambda_real_full[col, col] = eigenvalues[idx].real
            col += 1
        else:
            idx, partner = block[1], block[2]
            v = eigenvectors[:, idx]
            p, q = v.real, v.imag
            a, b = eigenvalues[idx].real, eigenvalues[idx].imag
            Q_real[:, col] = p
            Q_real[:, col + 1] = q
            Lambda_real_full[col, col] = a
            Lambda_real_full[col, col + 1] = b
            Lambda_real_full[col + 1, col] = -b
            Lambda_real_full[col + 1, col + 1] = a
            col += 2

    Q_real_inv = np.linalg.inv(Q_real)

    def reconstruct_evd(k):
        Lambda_k = np.zeros((n, n))
        covered = 0
        for b, start in zip(blocks, block_start):
            if covered >= k:
                break
            size = 1 if b[0] == "real" else 2
            if b[0] == "real":
                Lambda_k[start, start] = Lambda_real_full[start, start]
            else:
                Lambda_k[start : start + 2, start : start + 2] = (
                    Lambda_real_full[start : start + 2, start : start + 2]
                )
            covered += size
        A_k = Q_real @ Lambda_k @ Q_real_inv
        return A_k

    U, S, Vt = np.linalg.svd(A)

    def reconstruct_svd(k):
        Sigma_k = np.zeros((n, n))
        for i in range(k):
            Sigma_k[i, i] = S[i]
        A_k = U @ Sigma_k @ Vt
        return A_k

    def effective_k(k):
        covered = 0
        for b in blocks:
            if covered >= k:
                break
            covered += 1 if b[0] == "real" else 2
        return covered

    os.makedirs(output_dir, exist_ok=True)

    k_values = [50, 100, 160]

    for k in k_values:
        k_actual = effective_k(k)
        if k_actual != k:
            print(
                f"Using k={k_actual} for EVD instead of k = {k} to avoid complex conjugate eigenvalues split."
            )

        A_evd = reconstruct_evd(k)
        A_svd = reconstruct_svd(k)

        error_evd = np.abs(A - A_evd)
        error_svd = np.abs(A - A_svd)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        axes[0, 0].imshow(A, cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title("Original")
        axes[0, 1].imshow(A_evd, cmap="gray", vmin=0, vmax=255)
        axes[0, 1].set_title(f"EVD Reconstruction (k={k})")
        axes[0, 2].imshow(error_evd, cmap="gray")
        axes[0, 2].set_title("EVD Error |A - A_k|")

        axes[1, 0].imshow(A, cmap="gray", vmin=0, vmax=255)
        axes[1, 0].set_title("Original")
        axes[1, 1].imshow(A_svd, cmap="gray", vmin=0, vmax=255)
        axes[1, 1].set_title(f"SVD Reconstruction (k={k})")
        axes[1, 2].imshow(error_svd, cmap="gray")
        axes[1, 2].set_title("SVD Error |A - A_k|")

        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"task1_k{k}.png"), dpi=150)
        plt.close()

        frob_evd = np.linalg.norm(A - A_evd, ord="fro")
        frob_svd = np.linalg.norm(A - A_svd, ord="fro")
        print(
            f"k={k}: Frobenius error EVD = {frob_evd:.2f}, SVD = {frob_svd:.2f}"
        )
    k_range = range(1, n + 1)
    errors_evd = []
    errors_svd = []

    for k in k_range:
        A_evd = reconstruct_evd(k)
        A_svd = reconstruct_svd(k)
        errors_evd.append(np.linalg.norm(A - A_evd, ord="fro"))
        errors_svd.append(np.linalg.norm(A - A_svd, ord="fro"))

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, errors_evd, label="EVD reconstruction error")
    plt.plot(k_range, errors_svd, label="SVD reconstruction error")
    plt.xlabel("Number of retained components (k)")
    plt.ylabel("Frobenius norm error E(k)")
    plt.title("Reconstruction error vs k (Square image)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "task1_error_vs_k.png"), dpi=150)
    plt.close()

    np.save(
        os.path.join(output_dir, "task1_errors_evd.npy"), np.array(errors_evd)
    )
    np.save(
        os.path.join(output_dir, "task1_errors_svd.npy"), np.array(errors_svd)
    )

    print(f"Task 1 done. Figures saved in {output_dir}/")


def run_task2(data_dir="data", output_dir="figures"):
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


def main():
    parser = argparse.ArgumentParser(
        description="Assignment 2: EVD/SVD image reconstruction"
    )
    parser.add_argument(
        "--task",
        choices=["1", "2", "both"],
        default="both",
        help="Which task to run (default: both)",
    )
    parser.add_argument(
        "--data_dir", default="data", help="Folder containing the input images"
    )
    parser.add_argument(
        "--output_dir",
        default="figures",
        help="Folder to write output figures to",
    )
    args = parser.parse_args()

    if args.task in ("1", "both"):
        print("=== Task 1: EVD and SVD on square image ===")
        run_task1(data_dir=args.data_dir, output_dir=args.output_dir)
        print()

    if args.task in ("2", "both"):
        print("=== Task 2: SVD on rectangular image ===")
        run_task2(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
