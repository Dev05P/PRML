import random
import math
import matplotlib.pyplot as plt


def generateGaussianDataset(mu, sigma2, numSamples):
    sigma = math.sqrt(sigma2)
    dataset = []
    while len(dataset) < numSamples:
        u1 = random.uniform(-1, 1)
        u2 = random.uniform(-1, 1)
        s = u1**2 + u2**2
        if 0 < s < 1:
            k = math.sqrt((-2 * math.log(s)) / s)
            dataset.append(mu + sigma * (u1 * k))
            if len(dataset) < numSamples:
                dataset.append(mu + sigma * (u2 * k))

    return dataset


if __name__ == "__main__":
    print("2(c) Gaussian Histogram Plotting")
    mu = float(input("Enter mean (μ): "))
    sigma2 = float(input("Enter variance (σ²): "))
    numSamples = int(input("Enter number of samples: "))
    gaussianSamples = generateGaussianDataset(mu, sigma2, numSamples)
    plt.figure(figsize=(8, 5))
    plt.hist(gaussianSamples, bins=50, color='salmon',
             edgecolor='black', density=True)
    plt.title(f"Gaussian Distribution N({mu}, {sigma2})")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.grid(True, alpha=0.3)
    plt.show()
