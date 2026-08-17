import random
import math


def generateGaussianPair(mu, sigma2):
    sigma = math.sqrt(sigma2)
    while True:
        u1 = random.uniform(-1, 1)
        u2 = random.uniform(-1, 1)
        s = u1**2 + u2**2
        if 0 < s < 1:
            k = math.sqrt((-2 * math.log(s)) / s)
            x = u1 * k
            y = u2 * k
            xPrime = mu + sigma * x
            yPrime = mu + sigma * y
            return xPrime, yPrime


if __name__ == "__main__":
    print("2(b) Gaussian Variable Generator")
    mu = float(input("Enter mean (μ): "))
    sigma2 = float(input("Enter variance (σ²): "))

    xVal, yVal = generateGaussianPair(mu, sigma2)
    print(f"\nGenerated Gaussian Samples for N({mu}, {sigma2}):")
    print(f"Sample 1 (x'): {xVal}")
    print(f"Sample 2 (y'): {yVal}")
