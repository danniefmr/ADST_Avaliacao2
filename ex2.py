import matplotlib.pyplot as plt
import numpy as np


def simulate_ddos(
    N: int, lam: float, alpha: float, mu: float, tau: float, seed: int = 42
):
    rng = np.random.default_rng(seed)
    d = mu + (1.0 - mu) * tau

    states = np.zeros(N, dtype=int)
    current_state = 0

    leg_arrivals = 0
    leg_rejected = 0

    # Sorteios vetoriais para velocidade em N = 10^6 slots
    u_rem = rng.random(N)
    u_arr = rng.random(N)

    for t in range(N):
        # 1. Tentativa de remoção se o sistema não estiver vazio
        if current_state > 0 and u_rem[t] < d:
            current_state -= 1

        # 2. Chegada de pedido legítimo ou malicioso (mutuamente exclusivos)
        if u_arr[t] < lam:
            leg_arrivals += 1
            if current_state < 3:
                current_state += 1
            else:
                leg_rejected += 1
        elif u_arr[t] < lam + alpha:
            if current_state < 3:
                current_state += 1

        states[t] = current_state

    pi_sim = np.bincount(states, minlength=4) / N
    rej_cond_sim = leg_rejected / leg_arrivals if leg_arrivals > 0 else 0.0
    rej_joint_sim = leg_rejected / N
    return states, pi_sim, rej_cond_sim, rej_joint_sim


def theoretical_pi(lam: float, alpha: float, mu: float, tau: float) -> np.ndarray:
    d = mu + (1.0 - mu) * tau
    p_ch = lam + alpha
    ratio_1_0 = p_ch / (d * (1.0 - p_ch))
    ratio_step = ((1.0 - d) * p_ch) / (d * (1.0 - p_ch))

    w = np.zeros(4)
    w[0] = 1.0
    w[1] = ratio_1_0
    w[2] = w[1] * ratio_step
    w[3] = w[2] * ratio_step
    return w / np.sum(w)


def main():
    N = 1_000_000
    lam, mu, tau = 0.2, 0.3, 0.2
    d = mu + (1.0 - mu) * tau
    alphas = [0.5, 0.75]

    results = {}
    for idx, alpha in enumerate(alphas):
        states, pi_sim, rej_cond_sim, rej_joint_sim = simulate_ddos(
            N=N, lam=lam, alpha=alpha, mu=mu, tau=tau, seed=100 + idx
        )
        pi_theo = theoretical_pi(lam=lam, alpha=alpha, mu=mu, tau=tau)
        rej_cond_theo = pi_theo[3] * (1.0 - d)
        rej_joint_theo = lam * rej_cond_theo

        results[alpha] = (
            pi_sim,
            pi_theo,
            rej_cond_sim,
            rej_cond_theo,
            rej_joint_sim,
            rej_joint_theo,
            states,
        )

        print(f"=== Cenário alpha = {alpha} ===")
        for s in range(4):
            print(
                f"  Estado {s}: Sim = {pi_sim[s]:.6f} | Teórico = {pi_theo[s]:.6f}"
            )
        print(
            f"  Rejeição condicional legítima: Sim = {rej_cond_sim:.6f} | Teórico = {rej_cond_theo:.6f}"
        )
        print(
            f"  Rejeição conjunta (por slot):   Sim = {rej_joint_sim:.6f} | Teórico = {rej_joint_theo:.6f}\n"
        )

    # Mostra os primeiros 100 slots para manter a evolução legível.
    n_slots_plot = min(100, N)
    slots = np.arange(1, n_slots_plot + 1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, alpha in zip(axes, alphas):
        ax.step(
            slots,
            results[alpha][6][:n_slots_plot],
            where="post",
            linewidth=0.8,
        )
        ax.set_title(f"Ocupação nos primeiros 100 slots (α = {alpha:.2f})")
        ax.set_ylabel("Pedidos no sistema")
        ax.set_yticks(range(4))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("Slot")
    fig.tight_layout()
    plt.show()

    # Gráfico comparativo das distribuições estacionárias
    estados = np.arange(4)
    largura = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        estados - 1.5 * largura,
        results[0.5][1],
        width=largura,
        label="Teórico (α = 0.50)",
    )
    ax.bar(
        estados - 0.5 * largura,
        results[0.5][0],
        width=largura,
        label="Simulado (α = 0.50)",
    )
    ax.bar(
        estados + 0.5 * largura,
        results[0.75][1],
        width=largura,
        label="Teórico (α = 0.75)",
    )
    ax.bar(
        estados + 1.5 * largura,
        results[0.75][0],
        width=largura,
        label="Simulado (α = 0.75)",
    )

    ax.set_title("Distribuição Estacionária: Teórica vs. Simulada (N = 10^6 slots)")
    ax.set_xlabel("Estado (Número de pedidos no sistema)")
    ax.set_ylabel("Probabilidade")
    ax.set_xticks(estados)
    ax.set_xticklabels(["0", "1", "2", "3"])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()