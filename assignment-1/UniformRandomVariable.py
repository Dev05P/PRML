import random
import matplotlib.pyplot as plt


def generateUniform(a, b, numSamples):
    return [a + (b - a) * random.random() for _ in range(numSamples)]


if __name__ == "__main__":
    print("2(a) Uniform Random Variables")
    a = float(input("Enter lower bound a: "))
    b = float(input("Enter upper bound b: "))
    numSamples = int(input("Enter number of samples: "))
    samples = generateUniform(a, b, numSamples)
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=50, color='skyblue',
             edgecolor='black', density=True)
    plt.title(f"Uniform Distribution U({a}, {b})")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.grid(True, alpha=0.3)
    plt.show()
