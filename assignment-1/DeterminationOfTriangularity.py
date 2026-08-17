def isUpperTriangular(A):
    m, n = len(A), len(A[0])
    if m != n:
        return False
    for i in range(1, m):
        for j in range(0, i):
            if A[i][j] != 0:
                return False
    return True


def isLowerTriangular(A):
    m, n = len(A), len(A[0])
    if m != n:
        return False
    for i in range(0, m):
        for j in range(i + 1, n):
            if A[i][j] != 0:
                return False
    return True


if __name__ == "__main__":
    print("1(e) Triangular Matrix Check")
    m = int(input("Enter no. of rows: "))
    n = int(input("Enter no. of cols: "))

    print("\nEnter Matrix:")
    A = [list(map(float, input(f"Row {i + 1}: ").split())) for i in range(m)]

    upper = isUpperTriangular(A)
    lower = isLowerTriangular(A)

    print(f"\nIs Upper Triangular: {upper}")
    print(f"Is Lower Triangular: {lower}")
