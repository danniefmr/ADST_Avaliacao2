import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


mu = 125  # taxa de serviço, em pedidos por segundo
capacity = 5
lambdas = np.arange(10, 201)  # 10 a 200 pedidos por segundo

rho = lambdas / mu
states = np.arange(capacity + 1)
stationary_weights = rho[:, np.newaxis] ** states
stationary_probabilities = stationary_weights / stationary_weights.sum(
    axis=1, keepdims=True
)
blocking_probability = stationary_probabilities[:, capacity]

lambda_reference = 80
reference_index = np.flatnonzero(lambdas == lambda_reference)[0]
blocking_reference = blocking_probability[reference_index]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(lambdas, blocking_probability, color="tab:blue")
ax.scatter(
    lambda_reference,
    blocking_reference,
    color="tab:red",
    zorder=3,
    label=f"λ = 80: P_bloqueio = {blocking_reference:.2%}",
)
ax.axvline(mu, color="tab:gray", linestyle="--", label="λ = μ")
ax.set_title("Probabilidade de bloqueio no modelo M/M/1/K (K = 5)")
ax.set_xlabel("Taxa de chegada, λ (pedidos/s)")
ax.set_ylabel("Probabilidade de bloqueio")
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_ylim(0, 1)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
plt.show()