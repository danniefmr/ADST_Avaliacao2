import numpy as np
import matplotlib.pyplot as plt

# Reprodutibilidade
rng = np.random.default_rng(42)

# Parâmetros (Dica 1)
lam = 9.0
lam_l = 2.0
lam_h = 10.0
mu_h = 1.3
mu_l = 0.2
T = 100_000
n_realizations = 1000


def simulate_poisson(T, lam, rng):
  return rng.poisson(lam=lam, size=T)


def simulate_on_off(T, lam_l, lam_h, mu_h, mu_l, rng):
  # Gera sequência de estados (True = ON, False = OFF) até atingir T
  states = []
  is_on = bool(rng.integers(0, 2))  # início aleatório ON ou OFF

  while len(states) < T:
    scale = (1.0 / mu_h) if is_on else (1.0 / mu_l)
    dur = int(np.round(rng.exponential(scale=scale)))
    states.extend([is_on] * dur)
    is_on = not is_on

  states = np.array(states[:T], dtype=bool)
  rates = np.where(states, lam_h, lam_l)
  return rng.poisson(lam=rates)


# 1(a) Simulação de uma realização longa
x_poisson = simulate_poisson(T, lam, rng)
x_onoff = simulate_on_off(T, lam_l, lam_h, mu_h, mu_l, rng)

# 1(b) Intervalos não sobrepostos
n_blocks = 50
block_size = T // n_blocks
blocks_idx = np.arange(1, n_blocks + 1)

p_blocks = x_poisson.reshape(n_blocks, block_size)
o_blocks = x_onoff.reshape(n_blocks, block_size)

p_means = p_blocks.mean(axis=1)
p_vars = p_blocks.var(axis=1, ddof=1)

o_means = o_blocks.mean(axis=1)
o_vars = o_blocks.var(axis=1, ddof=1)

# Figuras
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# Poisson - Média
axes[0, 0].plot(blocks_idx, p_means, marker="o", markersize=3)
axes[0, 0].axhline(
    lam, color="r", linestyle="--", label=f"Teórico ({lam:.2f})"
)
axes[0, 0].set_title("Poisson: Média por intervalo")
axes[0, 0].set_ylabel("Pacotes / unidade de tempo")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Poisson - Variância
axes[0, 1].plot(blocks_idx, p_vars, marker="o", markersize=3, color="tab:orange")
axes[0, 1].axhline(
    lam, color="r", linestyle="--", label=f"Teórico Poisson ({lam:.2f})"
)
axes[0, 1].set_title("Poisson: Variância por intervalo")
axes[0, 1].set_ylabel("Variância")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# On-Off - Média
mean_onoff_approx = np.mean(o_means)
axes[1, 0].plot(blocks_idx, o_means, marker="o", markersize=3)
axes[1, 0].axhline(
    mean_onoff_approx,
    color="r",
    linestyle="--",
    label=f"Média empírica (~{mean_onoff_approx:.2f})",
)
axes[1, 0].set_title("On-Off: Média por intervalo")
axes[1, 0].set_xlabel("Intervalo não sobreposto")
axes[1, 0].set_ylabel("Pacotes / unidade de tempo")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# On-Off - Variância
var_onoff_approx = np.mean(o_vars)
axes[1, 1].plot(blocks_idx, o_vars, marker="o", markersize=3, color="tab:orange")
axes[1, 1].axhline(
    var_onoff_approx,
    color="r",
    linestyle="--",
    label=f"Var. empírica (~{var_onoff_approx:.2f})",
)
axes[1, 1].set_title("On-Off: Variância por intervalo")
axes[1, 1].set_xlabel("Intervalo não sobreposto")
axes[1, 1].set_ylabel("Variância")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


time_mean_poisson = np.mean(x_poisson)
time_mean_onoff = np.mean(x_onoff)

running_mean_poisson = np.cumsum(x_poisson) / np.arange(1, T + 1)
running_mean_onoff = np.cumsum(x_onoff) / np.arange(1, T + 1)

T_ens = 100000 

ensemble_poisson = np.array(
    [simulate_poisson(T_ens, lam, rng) for _ in range(n_realizations)]
)
ensemble_onoff = np.array([
    simulate_on_off(T_ens, lam_l, lam_h, mu_h, mu_l, rng)
    for _ in range(n_realizations)
])

# Média ensemble em cada instante t (média ao longo das 1000 realizações)
ens_mean_t_poisson = ensemble_poisson.mean(axis=0)
ens_mean_t_onoff = ensemble_onoff.mean(axis=0)

# Média global do ensemble
ens_global_mean_poisson = ens_mean_t_poisson.mean()
ens_global_mean_onoff = ens_mean_t_onoff.mean()

# Imprimir comparação numérica
print("=== COMPARAÇÃO DE ERGODICIDADE (MÉDIA) ===")
print(f"Poisson -> Média Temporal (1 realiz., T={T}): {time_mean_poisson:.4f}")
print(
    f"Poisson -> Média Ensemble ({n_realizations} realiz.):"
    f" {ens_global_mean_poisson:.4f}"
)
print(
    f"Diferença absoluta Poisson: {abs(time_mean_poisson - ens_global_mean_poisson):.4f}\n"
)

print(f"On-Off  -> Média Temporal (1 realiz., T={T}): {time_mean_onoff:.4f}")
print(
    f"On-Off  -> Média Ensemble ({n_realizations} realiz.):"
    f" {ens_global_mean_onoff:.4f}"
)
print(
    f"Diferença absoluta On-Off: {abs(time_mean_onoff - ens_global_mean_onoff):.4f}"
)

# Visualização gráfica da convergência e comparação
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Gráfico 1: Poisson
axes[0].plot(
    ens_mean_t_poisson,
    alpha=0.5,
    label="Média Ensemble E[X(t)] (1000 realiz.)",
    color="steelblue",
)
axes[0].plot(
    running_mean_poisson[:T_ens],
    label="Média Temporal acumulada (1 realiz.)",
    color="darkorange",
    linewidth=2,
)
axes[0].axhline(
    time_mean_poisson,
    color="red",
    linestyle="--",
    label=f"Média Temporal final (T={T}): {time_mean_poisson:.3f}",
)
axes[0].set_title("Ergodicidade: Processo de Poisson")
axes[0].set_xlabel("Tempo (t)")
axes[0].set_ylabel("Taxa média de pacotes")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico 2: On-Off
axes[1].plot(
    ens_mean_t_onoff,
    alpha=0.5,
    label="Média Ensemble E[X(t)] (1000 realiz.)",
    color="steelblue",
)
axes[1].plot(
    running_mean_onoff[:T_ens],
    label="Média Temporal acumulada (1 realiz.)",
    color="darkorange",
    linewidth=2,
)
axes[1].axhline(
    time_mean_onoff,
    color="red",
    linestyle="--",
    label=f"Média Temporal final (T={T}): {time_mean_onoff:.3f}",
)
axes[1].set_title("Ergodicidade: Processo On-Off")
axes[1].set_xlabel("Tempo (t)")
axes[1].set_ylabel("Taxa média de pacotes")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- EXERCÍCIO 4: Autocorrelação e Estacionaridade em Sentido Amplo (WSS) ---


def autocorrelation_lags(x, max_lag=30):
  """Calcula R_X(tau) = E[X(t) * X(t + tau)] para tau = 0, 1, ..., max_lag."""
  N = len(x)
  r = np.empty(max_lag + 1)
  for tau in range(max_lag + 1):
    r[tau] = np.mean(x[: N - tau] * x[tau:])
  return r


max_lag = 30
lags = np.arange(max_lag + 1)

# Dividimos a série em duas metades temporais (t1 e t2) para verificar
# se R_X(t, t+tau) depende apenas de tau e não do instante absoluto t
mid = T // 2

R_p_1 = autocorrelation_lags(x_poisson[:mid], max_lag)
R_p_2 = autocorrelation_lags(x_poisson[mid:], max_lag)

R_o_1 = autocorrelation_lags(x_onoff[:mid], max_lag)
R_o_2 = autocorrelation_lags(x_onoff[mid:], max_lag)

# Visualização
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Gráfico 1: Poisson
axes[0].plot(lags, R_p_1, "o-", label="1.ª metade do tempo", alpha=0.8)
axes[0].plot(lags, R_p_2, "x--", label="2.ª metade do tempo", alpha=0.8)
axes[0].axhline(
    lam**2, color="r", linestyle=":", label=f"E[X]² = λ² = {lam**2:.1f}"
)
axes[0].set_title("Autocorrelacão R_X(τ) - Processo de Poisson")
axes[0].set_xlabel("Desfasamento τ (lag)")
axes[0].set_ylabel("R_X(τ) = E[X(t)X(t+τ)]")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico 2: On-Off
axes[1].plot(lags, R_o_1, "o-", label="1.ª metade do tempo", alpha=0.8)
axes[1].plot(lags, R_o_2, "x--", label="2.ª metade do tempo", alpha=0.8)
axes[1].axhline(
    np.mean(x_onoff) ** 2,
    color="r",
    linestyle=":",
    label=f"E[X]² ≈ {np.mean(x_onoff)**2:.2f}",
)
axes[1].set_title("Autocorrelacão R_X(τ) - Processo On-Off")
axes[1].set_xlabel("Desfasamento τ (lag)")
axes[1].set_ylabel("R_X(τ) = E[X(t)X(t+τ)]")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()