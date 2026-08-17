def matrixMultiply(A, B):
    m, n = len(A), len(A[0])
    k = len(B[0])
    result = [[0 for _ in range(k)] for _ in range(m)]
    for i in range(m):
        for j in range(k):
            for x in range(n):
                result[i][j] += A[i][x] * B[x][j]

    return result


if __name__ == "__main__":
    print("1(a) Matrix Multiplication")
    m = int(input("Enter no. of rows for Matrix A: "))
    n = int(input("Enter no. of cols for Matrix A, no. of rows for Matrix B: "))
    k = int(input("Enter no. of cols for Matrix B: "))

    print("\nEnter Matrix A:")
    A = [list(map(float, input(f"Row {i + 1}: ").split())) for i in range(m)]

    print("\nEnter Matrix B:")
    B = [list(map(float, input(f"Row {i + 1}: ").split())) for i in range(n)]

    result = matrixMultiply(A, B)
    print("\nResulting Matrix (m x k):")
    for row in result:
        print(row)
