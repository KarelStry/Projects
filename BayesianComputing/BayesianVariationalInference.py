import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Simulated data
np.random.seed(42)
data = np.random.normal(5, 2, 100)

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=data)

    approx = pm.fit(n=50000, method='advi')
    plt.figure(figsize=(10, 4))
    plt.plot(approx.hist, label="ELBO")
    plt.title("ELBO Convergence During ADVI")
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    trace = approx.sample(1000)
    

# Extract variational samples
mu_samples = trace.posterior["mu"].values.flatten()
sigma_samples = trace.posterior["sigma"].values.flatten()

print(approx.mean.eval())
print(approx.std.eval())

# Plotting with matplotlib
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(mu_samples, bins=30, density=True, alpha=0.7, color="skyblue", edgecolor="k")
plt.title("Posterior of μ")
plt.xlabel("mu")
plt.ylabel("Density")

plt.subplot(1, 2, 2)
plt.hist(sigma_samples, bins=30, density=True, alpha=0.7, color="salmon", edgecolor="k")
plt.title("Posterior of σ")
plt.xlabel("sigma")
plt.ylabel("Density")

plt.tight_layout()

plt.show()
