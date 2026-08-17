def dotProduct(v1, v2):
    return sum(v1[i] * v2[i] for i in range(len(v1)))


if __name__ == "__main__":
    print("1(b) Vector Dot Product")
    n = int(input("Enter dimension n for vectors: "))
    v1 = list(
        map(float, input(f"Enter Vector 1 ({n} numbers space-separated): ").split()))
    v2 = list(
        map(float, input(f"Enter Vector 2 ({n} numbers space-separated): ").split()))
    if len(v1) != n or len(v2) != n:
        print("Error: Vector dimension mismatch!")
    else:
        print(f"\nDot Product: {dotProduct(v1, v2)}")
