def isSymmetric(A):
    m, n = len(A), len(A[0])
    if m != n:
        return False
    for i in range(m):
        for j in range(i + 1, n):
            if A[i][j] != A[j][i]:
                return False

    return True


if __name__ == "__main__":
    print("1(d) Symmetric Matrix Check")
    m = int(input("Enter rows (m): "))
    n = int(input("Enter cols (n): "))

    print("\nEnter Matrix:")
    A = [list(map(float, input(f"Row {i + 1}: ").split())) for i in range(m)]

    if isSymmetric(A):
        print("\nThe matrix is symmetric.")
    else:
        print("\nThe matrix is not symmetric.")
