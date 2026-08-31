import math
import random
import matplotlib.pyplot as plt


def readData(path):
    xs = []
    ys = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
    return xs, ys


def mergeSort(values):
    if len(values) <= 1:
        return values[:]
    mid = len(values) // 2
    left = mergeSort(values[:mid])
    right = mergeSort(values[mid:])
    merged = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def splitData(xs, ys, trainFrac, testFrac, seed):
    n = len(xs)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    nTrain = int(n * trainFrac)
    nTest = int(n * testFrac)
    trainIdx = idx[:nTrain]
    testIdx = idx[nTrain:nTrain + nTest]
    valIdx = idx[nTrain + nTest:]
    xTrain = [xs[i] for i in trainIdx]
    yTrain = [ys[i] for i in trainIdx]
    xTest = [xs[i] for i in testIdx]
    yTest = [ys[i] for i in testIdx]
    xVal = [xs[i] for i in valIdx]
    yVal = [ys[i] for i in valIdx]
    return xTrain, yTrain, xTest, yTest, xVal, yVal


def designRow(x, degree):
    row = []
    p = 1.0
    for _ in range(degree + 1):
        row.append(p)
        p *= x
    return row


def designMatrix(xs, degree):
    return [designRow(x, degree) for x in xs]


def transpose(m):
    return [list(row) for row in zip(*m)]


def matMul(a, b):
    n, k = len(a), len(a[0])
    m = len(b[0])
    result = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for t in range(k):
            aVal = a[i][t]
            for j in range(m):
                result[i][j] += aVal * b[t][j]
    return result


def matVec(a, v):
    return [sum(a[i][j] * v[j] for j in range(len(v))) for i in range(len(a))]


def solveLinearSystem(a, b):
    n = len(a)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivotRow = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivotRow] = aug[pivotRow], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            pivot = 1e-12
        for j in range(col, n + 1):
            aug[col][j] /= pivot
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                for j in range(col, n + 1):
                    aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def fitPolynomial(xs, ys, degree):
    xMat = designMatrix(xs, degree)
    xT = transpose(xMat)
    xtx = matMul(xT, xMat)
    xty = matVec(xT, ys)
    weights = solveLinearSystem(xtx, xty)
    return weights


def predict(weights, xs):
    degree = len(weights) - 1
    preds = []
    for x in xs:
        row = designRow(x, degree)
        preds.append(sum(w * r for w, r in zip(weights, row)))
    return preds


def computeMse(yTrue, yPred):
    n = len(yTrue)
    return sum((a - b) ** 2 for a, b in zip(yTrue, yPred)) / n


def plotDegreeVsError(degrees, trainErrs, testErrs, chosenDegree):
    plt.figure(figsize=(7, 5))
    plt.plot(degrees, trainErrs, marker="o", label="Train MSE")
    plt.plot(degrees, testErrs, marker="o", label="Test MSE")
    plt.axvline(chosenDegree, color="gray", linestyle="--", label=f"chosen degree = {chosenDegree}")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Mean squared error")
    plt.title("Error vs polynomial degree")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("degree_vs_error.png", dpi=150)
    plt.close()


def plotFittedCurve(xs, ys, xSorted, weights, chosenDegree):
    yFit = predict(weights, xSorted)
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=4, alpha=0.25, label="Noisy observations")
    plt.plot(xSorted, yFit, color="red", linewidth=2, label=f"Fitted polynomial (degree {chosenDegree})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Polynomial regression fit")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fitted_curve.png", dpi=150)
    plt.close()


def classifyFit(trainMse, testMse, chosenTrainMse, chosenTestMse, threshold=0.01):
    if math.isnan(trainMse) or math.isnan(testMse):
        return "solver diverged (numerical overflow)"
    trainBetter = trainMse < chosenTrainMse * (1 - threshold)
    trainWorse = trainMse > chosenTrainMse * (1 + threshold)
    testBetter = testMse < chosenTestMse * (1 - threshold)
    testWorse = testMse > chosenTestMse * (1 + threshold)
    if trainBetter and testWorse:
        return "overfit"
    if trainWorse and testWorse:
        return "underfit"
    if testBetter:
        return "better than chosen fit"
    return "excess complexity - no gain"


def plotUnderfitOverfitComparison(xs, ys, xSorted, degreesToShow, resultsByDegree, chosenDegree, chosenTrainMse, chosenTestMse):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, degree in zip(axes, degreesToShow):
        trainMse, testMse, w = resultsByDegree[degree]
        yFit = predict(w, xSorted)
        label = "chosen fit" if degree == chosenDegree else classifyFit(trainMse, testMse, chosenTrainMse, chosenTestMse)
        ax.scatter(xs, ys, s=3, alpha=0.15)
        ax.plot(xSorted, yFit, color="red", linewidth=2)
        ax.set_title(f"degree {degree} ({label})")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("y")
    plt.tight_layout()
    plt.savefig("underfit_overfit_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    xs, ys = readData("noisy_21.txt")
    xTrain, yTrain, xTest, yTest, xVal, yVal = splitData(xs, ys, 0.6, 0.2, seed=42)

    results = []
    for degree in range(1, 76):
        w = fitPolynomial(xTrain, yTrain, degree)
        trainPred = predict(w, xTrain)
        testPred = predict(w, xTest)
        trainMse = computeMse(yTrain, trainPred)
        testMse = computeMse(yTest, testPred)
        results.append((degree, trainMse, testMse, w))
        print(f"degree {degree:2d}  trainMse {trainMse:12.4f}  testMse {testMse:12.4f}")

    argminDegree = min(results, key=lambda r: r[2])[0]

    elbowDegree = results[-1][0]
    for i in range(1, len(results)):
        remaining = results[i:]
        prevTestMse = results[i - 1][2]
        staysFlat = all(
            (prevTestMse - r[2]) / prevTestMse < 0.01 for r in remaining
        )
        if staysFlat:
            elbowDegree = results[i - 1][0]
            break

    chosenDegree = elbowDegree
    chosenResult = [r for r in results if r[0] == chosenDegree][0]
    chosenTrainMse = chosenResult[1]
    chosenTestMse = chosenResult[2]
    chosenW = chosenResult[3]
    valPred = predict(chosenW, xVal)
    valMse = computeMse(yVal, valPred)

    print()
    print(f"lowest test mse at degree = {argminDegree}")
    print(f"chosen degree (elbow)     = {chosenDegree}")
    print(f"validation mse            = {valMse:.4f}")
    print("weights:", chosenW)

    overfitLabels = {r[0]: classifyFit(r[1], r[2], chosenTrainMse, chosenTestMse) for r in results}
    overfitDegree = next((d for d, label in overfitLabels.items() if label == "overfit"), None)
    divergeDegree = next((d for d, label in overfitLabels.items() if label == "solver diverged (numerical overflow)"), None)

    print()
    print("checked degrees 1-75 for genuine overfitting:")
    if overfitDegree is None:
        print("no degree in this range classified as overfit")
    else:
        print(f"overfitting found at degree {overfitDegree}")
    if divergeDegree is not None:
        print(f"solver starts diverging (numerical overflow) at degree {divergeDegree}")

    with open("results.txt", "w") as f:
        f.write("degree,train_mse,test_mse,label\n")
        for degree, trainMseV, testMseV, _ in results:
            f.write(f"{degree},{trainMseV},{testMseV},{overfitLabels[degree]}\n")
        f.write(f"\nargmin_degree,{argminDegree}\n")
        f.write(f"chosen_degree,{chosenDegree}\n")
        f.write(f"val_mse,{valMse}\n")
        f.write(f"overfit_degree_found,{overfitDegree}\n")
        f.write(f"solver_diverges_at_degree,{divergeDegree}\n")
        f.write("weights," + ",".join(str(v) for v in chosenW) + "\n")

    resultsByDegree = {r[0]: (r[1], r[2], r[3]) for r in results}

    degrees = [r[0] for r in results]
    trainErrs = [r[1] for r in results]
    testErrs = [r[2] for r in results]
    plotDegreeVsError(degrees, trainErrs, testErrs, chosenDegree)

    xSorted = mergeSort(xs)
    plotFittedCurve(xs, ys, xSorted, chosenW, chosenDegree)

    degreesToShow = [2, chosenDegree, 16]
    plotUnderfitOverfitComparison(xs, ys, xSorted, degreesToShow, resultsByDegree, chosenDegree, chosenTrainMse, chosenTestMse)

    print("saved degree_vs_error.png, fitted_curve.png, underfit_overfit_comparison.png")
