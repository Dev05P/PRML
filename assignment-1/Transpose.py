def transpose(A):
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


if __name__ == "__main__":
    print("1(c) Matrix Transpose")
    m = int(input("Enter no. of rows: "))
    n = int(input("Enter no. of cols: "))

    print("\nEnter Matrix:")
    A = [list(map(float, input(f"Row {i + 1}: ").split())) for i in range(m)]

    A_T = transpose(A)
    print("\nTransposed Matrix (n x m):")
    for row in A_T:
        print(row)
