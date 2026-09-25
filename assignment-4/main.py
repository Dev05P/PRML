import math
import random
import matplotlib.pyplot as plt

random.seed(26)


def generateStandardNormal(count):
    values = []
    while len(values) < count:
        u = 2 * random.random() - 1
        v = 2 * random.random() - 1
        s = u * u + v * v
        if s >= 1 or s == 0:
            continue
        factor = math.sqrt(-2 * math.log(s) / s)
        values.append(u * factor)
        values.append(v * factor)
    return values[:count]


def generateIsotropic(count, mean1, mean2):
    d1 = generateStandardNormal(count)
    d2 = generateStandardNormal(count)
    points = []
    for i in range(count):
        points.append([mean1 + d1[i], mean2 + d2[i]])
    return points


def sampleMean(points):
    total1 = 0
    total2 = 0
    for p in points:
        total1 += p[0]
        total2 += p[1]
    return [total1 / len(points), total2 / len(points)]


def sampleCovariance(points, mean):
    s11 = 0
    s12 = 0
    s22 = 0
    for p in points:
        dx = p[0] - mean[0]
        dy = p[1] - mean[1]
        s11 += dx * dx
        s12 += dx * dy
        s22 += dy * dy
    n = len(points) - 1
    return [[s11 / n, s12 / n], [s12 / n, s22 / n]]


def eigenSymmetric(c):
    a = c[0][0]
    b = c[0][1]
    d = c[1][1]
    center = (a + d) / 2
    radius = math.sqrt(((a - d) / 2) ** 2 + b * b)
    lambda1 = center + radius
    lambda2 = center - radius
    theta = 0.5 * math.atan2(2 * b, a - d)
    vector1 = [math.cos(theta), math.sin(theta)]
    vector2 = [-math.sin(theta), math.cos(theta)]
    return lambda1, lambda2, vector1, vector2


def matrixSqrt(c):
    lambda1, lambda2, v1, v2 = eigenSymmetric(c)
    root1 = math.sqrt(lambda1)
    root2 = math.sqrt(lambda2)
    result = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            result[i][j] = root1 * v1[i] * v1[j] + root2 * v2[i] * v2[j]
    return result


def transformData(points, mean, root):
    transformed = []
    for p in points:
        dx = p[0] - mean[0]
        dy = p[1] - mean[1]
        transformed.append([
            mean[0] + root[0][0] * dx + root[0][1] * dy,
            mean[1] + root[1][0] * dx + root[1][1] * dy,
        ])
    return transformed


def determinant2(c):
    return c[0][0] * c[1][1] - c[0][1] * c[1][0]


def inverse2(c):
    det = determinant2(c)
    return [[c[1][1] / det, -c[0][1] / det], [-c[1][0] / det, c[0][0] / det]]


def mahalanobisSquared(px, py, mean, inv):
    dx = px - mean[0]
    dy = py - mean[1]
    return dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy)


def buildAxis(low, high, steps):
    stepSize = (high - low) / (steps - 1)
    axis = []
    for i in range(steps):
        axis.append(low + i * stepSize)
    return axis


def quadraticGrid(xs, ys, mean, inv):
    rows = []
    for y in ys:
        row = []
        for x in xs:
            row.append(mahalanobisSquared(x, y, mean, inv))
        rows.append(row)
    return rows


def plotDensity(points, title, fileName):
    mean = sampleMean(points)
    cov = sampleCovariance(points, mean)
    lambda1, lambda2, v1, v2 = eigenSymmetric(cov)
    inv = inverse2(cov)

    xData = [p[0] for p in points]
    yData = [p[1] for p in points]
    xs = buildAxis(min(xData) - 0.5, max(xData) + 0.5, 200)
    ys = buildAxis(min(yData) - 0.5, max(yData) + 0.5, 200)
    z = quadraticGrid(xs, ys, mean, inv)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xData, yData, s=6, alpha=0.25, color="gray")
    curves = ax.contour(xs, ys, z, levels=[1, 4, 9], colors=["tab:green", "tab:orange", "tab:red"])
    ax.clabel(curves, fmt={1: "d = 1", 4: "d = 2", 9: "d = 3"})

    length1 = math.sqrt(lambda1)
    length2 = math.sqrt(lambda2)
    ax.annotate("", xy=(mean[0] + length1 * v1[0], mean[1] + length1 * v1[1]),
                xytext=(mean[0] - length1 * v1[0], mean[1] - length1 * v1[1]),
                arrowprops=dict(arrowstyle="<->", color="blue", lw=2))
    ax.annotate("", xy=(mean[0] + length2 * v2[0], mean[1] + length2 * v2[1]),
                xytext=(mean[0] - length2 * v2[0], mean[1] - length2 * v2[1]),
                arrowprops=dict(arrowstyle="<->", color="purple", lw=2))
    ax.plot([], [], color="blue", lw=2,
            label="major axis: eigenvalue %.3f, length %.3f" % (lambda1, 2 * length1))
    ax.plot([], [], color="purple", lw=2,
            label="minor axis: eigenvalue %.3f, length %.3f" % (lambda2, 2 * length2))
    ax.scatter(mean[0], mean[1], marker="x", color="black", s=60)

    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.savefig(fileName, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return mean, cov, lambda1, lambda2, v1


def reportComparison(name, trueCov, mean, cov, lambda1, lambda2, v1):
    trueLambda1, trueLambda2, trueV1, trueV2 = eigenSymmetric(trueCov)
    angle = math.degrees(math.atan2(v1[1], v1[0]))
    print(name)
    print("  estimated mean:", [round(mean[0], 3), round(mean[1], 3)])
    print("  true covariance:     ", trueCov)
    print("  estimated covariance:", [[round(cov[0][0], 3), round(cov[0][1], 3)],
                                      [round(cov[1][0], 3), round(cov[1][1], 3)]])
    print("  true eigenvalues:      %.3f  %.3f" % (trueLambda1, trueLambda2))
    print("  estimated eigenvalues: %.3f  %.3f" % (lambda1, lambda2))
    print("  estimated major axis angle: %.2f degrees" % angle)
    print()


def generateClass(count, mean, cov):
    base = generateIsotropic(count, mean[0], mean[1])
    return transformData(base, mean, matrixSqrt(cov))


def classParameters(points):
    mean = sampleMean(points)
    cov = sampleCovariance(points, mean)
    return [mean, inverse2(cov), determinant2(cov), cov]


def discriminant(px, py, params):
    return -0.5 * mahalanobisSquared(px, py, params[0], params[1]) - 0.5 * math.log(params[2])


def boundaryGrid(xs, ys, params1, params2):
    rows = []
    for y in ys:
        row = []
        for x in xs:
            row.append(discriminant(x, y, params1) - discriminant(x, y, params2))
        rows.append(row)
    return rows


def classificationAccuracy(class1, class2, params1, params2):
    correct = 0
    for p in class1:
        if discriminant(p[0], p[1], params1) > discriminant(p[0], p[1], params2):
            correct += 1
    for p in class2:
        if discriminant(p[0], p[1], params2) > discriminant(p[0], p[1], params1):
            correct += 1
    return 100 * correct / (len(class1) + len(class2))


def plotBoundary(ax, class1, class2, title):
    params1 = classParameters(class1)
    params2 = classParameters(class2)
    accuracy = classificationAccuracy(class1, class2, params1, params2)

    allPoints = class1 + class2
    xData = [p[0] for p in allPoints]
    yData = [p[1] for p in allPoints]
    xs = buildAxis(min(xData) - 0.5, max(xData) + 0.5, 200)
    ys = buildAxis(min(yData) - 0.5, max(yData) + 0.5, 200)
    h = boundaryGrid(xs, ys, params1, params2)

    ax.contourf(xs, ys, h, levels=[-1e9, 0, 1e9], colors=["#cfe2f3", "#f4cccc"])
    ax.contour(xs, ys, h, levels=[0], colors="black", linewidths=2)
    ax.scatter([p[0] for p in class1], [p[1] for p in class1], s=8, color="tab:red", alpha=0.6, label="class 1")
    ax.scatter([p[0] for p in class2], [p[1] for p in class2], s=8, color="tab:blue", alpha=0.6, label="class 2")
    ax.scatter(params1[0][0], params1[0][1], marker="x", color="black", s=70)
    ax.scatter(params2[0][0], params2[0][1], marker="x", color="black", s=70)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("%s\naccuracy %.1f%%" % (title, accuracy), fontsize=10)
    ax.legend(loc="upper left", fontsize=8)

    print(title)
    print("  class 1 mean:", [round(params1[0][0], 3), round(params1[0][1], 3)])
    print("  class 2 mean:", [round(params2[0][0], 3), round(params2[0][1], 3)])
    print("  class 1 covariance:", [[round(params1[3][0][0], 3), round(params1[3][0][1], 3)],
                                    [round(params1[3][1][0], 3), round(params1[3][1][1], 3)]])
    print("  class 2 covariance:", [[round(params2[3][0][0], 3), round(params2[3][0][1], 3)],
                                    [round(params2[3][1][0], 3), round(params2[3][1][1], 3)]])
    print("  accuracy: %.1f%%" % accuracy)
    print()


def main():
    count = 1000
    mu = [2, 3]

    x = generateIsotropic(count, mu[0], mu[1])
    isoCov = [[1, 0], [0, 1]]
    mean, cov, l1, l2, v1 = plotDensity(x, "Isotropic covariance, C = I", "isotropic.png")
    reportComparison("Problem 1: isotropic", isoCov, mean, cov, l1, l2, v1)

    diagCov = [[4, 0], [0, 1]]
    yDiag = transformData(x, mu, matrixSqrt(diagCov))
    mean, cov, l1, l2, v1 = plotDensity(yDiag, "Diagonal covariance, C = diag(4, 1)", "diagonal.png")
    reportComparison("Problem 2: diagonal", diagCov, mean, cov, l1, l2, v1)

    fullCov = [[3, 1.8], [1.8, 2]]
    yFull = transformData(x, mu, matrixSqrt(fullCov))
    mean, cov, l1, l2, v1 = plotDensity(yFull, "Full covariance, C = [[3, 1.8], [1.8, 2]]", "full.png")
    reportComparison("Problem 3: full", fullCov, mean, cov, l1, l2, v1)

    classCount = 500
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    sharedCov = [[1.5, 0], [0, 1.5]]
    a1 = generateClass(classCount, [0, 0], sharedCov)
    a2 = generateClass(classCount, [4, 0], sharedCov)
    plotBoundary(axes[0][0], a1, a2, "A: equal isotropic covariances, 1.5 I")

    b1 = generateClass(classCount, [0, 0], fullCov)
    b2 = generateClass(classCount, [4, 0], fullCov)
    plotBoundary(axes[0][1], b1, b2, "B: equal full covariances")

    c1 = generateClass(classCount, [0, 0], [[1, 0], [0, 1]])
    c2 = generateClass(classCount, [4, 0], [[4, 1.5], [1.5, 2]])
    plotBoundary(axes[1][0], c1, c2, "C: different covariances, different means")

    d1 = generateClass(classCount, [2, 2], [[0.5, 0], [0, 0.5]])
    d2 = generateClass(classCount, [2, 2], [[4, 1.5], [1.5, 2]])
    plotBoundary(axes[1][1], d1, d2, "D: different covariances, same mean")

    fig.tight_layout()
    fig.savefig("boundaries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


main()
