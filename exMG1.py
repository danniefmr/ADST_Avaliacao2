import matplotlib.pyplot as plt
import numpy as np


mu = 125  # taxa de serviço, em pedidos por segundo
mean_service_time = 1 / mu
service_time_variance = 0.0001  # segundos^2
lambdas = np.arange(10, 125)  # 10 a 124 pedidos por segundo

rho = lambdas / mu
mean_queue = (
    rho**2
    / (2 * (1 - rho))
    * (1 + service_time_variance / mean_service_time**2)
)
mean_requests = mean_queue + rho
mean_wait = mean_queue / lambdas
mean_total_delay = mean_wait + mean_service_time

lambda_reference = 80
reference_index = np.flatnonzero(lambdas == lambda_reference)[0]
delay_reference = mean_total_delay[reference_index]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(lambdas, mean_total_delay, color="tab:blue")
ax.scatter(
    lambda_reference,
    delay_reference,
    color="tab:red",
    zorder=3,
    label=f"λ = 80: T = {delay_reference:.5f} s",
)
ax.set_title("Atraso médio total no modelo M/G/1")
ax.set_xlabel("Taxa de chegada, λ (pedidos/s)")
ax.set_ylabel("Atraso médio total, T (s)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
plt.show()