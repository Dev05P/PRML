import argparse

from src import task1_square_evd_svd
from src import task2_rectangular_svd


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
        task1_square_evd_svd.run(
            data_dir=args.data_dir, output_dir=args.output_dir
        )
        print()

    if args.task in ("2", "both"):
        print("=== Task 2: SVD on rectangular image ===")
        task2_rectangular_svd.run(
            data_dir=args.data_dir, output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
