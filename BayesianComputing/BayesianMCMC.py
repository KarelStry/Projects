import numpy as np
import matplotlib.pyplot as plt

# Simulated observed data (single point)
data = 3.0
likelihood_std = 1.0

# Prior: N(0, 1)
def log_prior(theta):
    return -0.5 * theta**2

# Likelihood: N(data | theta, likelihood_std^2)
def log_likelihood(theta):
    return -0.5 * ((data - theta) / likelihood_std)**2

# Unnormalized log posterior
def log_posterior(theta):
    return log_likelihood(theta) + log_prior(theta)

# Metropolis-Hastings MCMC
def metropolis_hastings(init_theta, n_samples, proposal_std):
    samples = []
    theta = init_theta
    for _ in range(n_samples):
        # Propose a new theta
        theta_new = np.random.normal(theta, proposal_std)
        
        # Calculate acceptance probability
        log_alpha = log_posterior(theta_new) - log_posterior(theta)
        alpha = np.exp(min(0, log_alpha))  # Clamp to avoid overflow
        
        # Accept or reject
        if np.random.rand() < alpha:
            theta = theta_new  # Accept
        samples.append(theta)
    
    return np.array(samples)

# Run the MCMC
samples = metropolis_hastings(init_theta=0.0, n_samples=10000, proposal_std=0.5)

# Plot the samples
plt.figure(figsize=(10, 4))
plt.hist(samples, bins=50, density=True, alpha=0.7, label='MCMC Samples')
x = np.linspace(-2, 6, 1000)
true_posterior = np.exp(-0.5 * ((data - x) / likelihood_std)**2 - 0.5 * x**2)
true_posterior /= np.trapz(true_posterior, x)  # Normalize
plt.plot(x, true_posterior, 'r-', label='True Posterior')
plt.title("Posterior Sampling using Metropolis-Hastings")
plt.xlabel(r'$\theta$')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
