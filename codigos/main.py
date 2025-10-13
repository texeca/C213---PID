from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import control as ctl  # "python-control"

def extrair_dados():
    """
    Carrega o .mat, extrai t,u,y, plota entrada e saída,
    estima y0, y_inf, K, acha t(28,3%) e t(63,2%) e calcula tau e theta (Smith).
    Retorna um dicionário com os principais resultados.
    """
    # --- Passo 1: carregar t (tempo), u (entrada) e y (saída) do .mat ---
    mat_path = r"C:\Users\texeca\Desktop\C213_projeto_pid\datasets\Dataset_Grupo10_c213.mat"
    data = loadmat(mat_path)  # carrega todas as variáveis do arquivo

    # Formato A: chaves 'tiempo', 'entrada', 'salida'
    t = np.asarray(data['tiempo']).ravel().astype(float)   # tempo (vetor 1D)
    u = np.asarray(data['entrada']).ravel().astype(float)  # entrada
    y = np.asarray(data['salida']).ravel().astype(float)   # saída


    # --- Passo 3: estimar y0, y_inf e K assumindo degrau em t=0 de 0->75 (Δu=75) ---
    # y0: mediana dos primeiros instantes (patamar inicial)
    # y_inf: mediana dos últimos instantes (patamar final)

    # 1) Pela diferença simples (primeiro e último ponto)
    deltaV_extremos = float(y[-1] - y[0])

    # 2) Pela diferença robusta (medianas nos patamares)
    y0    = float(np.median(y[: max(5, len(y)//50) ]))       # ~primeiros 2% das amostras
    y_inf = float(np.median(y[- max(20, len(y)//20) :]))     # ~últimos 5% das amostras
    deltaV_mediana = float(y_inf - y0)

    deltaU = 75.0  # assumido: degrau 0 -> 75 em t=0
    K = deltaV_mediana / deltaU

#    print(f"y0 (inicial)  = {y0:.4f}")
#    print(f"y_inf (final) = {y_inf:.4f}")
#    print(f"deltaU        = {deltaU:.4f}")
#    print(f"ΔV (extremos) = {deltaV_extremos:.4f}")
#    print(f"ΔV (medianas) = {deltaV_mediana:.4f}")
#    print(f"Ganho K       = {K:.6f}")

    # --- Passo 4: Smith – níveis-alvo (na unidade de y) e tempos correspondentes ---
    dy = y_inf - y0
    y_283 = y0 + 0.283 * dy
    y_632 = y0 + 0.632 * dy

    # tempos em que y(t) atinge 28,3% e 63,2% do caminho total
    # np.interp(x, xp, fp): devolve fp quando xp cruza x
    t_283 = float(np.interp(y_283, y, t))
    t_632 = float(np.interp(y_632, y, t))

#    print(f"t(28,3%) = {t_283:.3f} s")
#    print(f"t(63,2%) = {t_632:.3f} s")

    # --- Passo 5: metodo de Smith (tau e theta) ---
    tau   = 1.5 * (t_632 - t_283)
    theta = t_632 - tau

    print(f"K     = {K:.6f}")
    print(f"tau   = {tau:.6f}")
    print(f"theta = {theta:.6f}")

    # retorna tudo que será útil nos próximos passos (FOPDT e PID)
    return {
        "t": t, "u": u, "y": y,
        "y0": y0, "y_inf": y_inf, "dy": dy,
        "deltaU": deltaU, "K": K,
        "t_283": t_283, "t_632": t_632,
        "tau": tau, "theta": theta
    }

def fopdt_step_response(t, K, tau, theta, deltaU, y0):
    """
    Resposta prevista de um FOPDT ao degrau ΔU, no mesmo eixo de tempo t.
    """
    t = np.asarray(t, dtype=float)
    y_hat = np.full_like(t, fill_value=y0, dtype=float)   # antes do atraso: y0
    idx = t >= theta                                      # após o atraso: exponencial
    if tau <= 0:
        y_hat[idx] = y0 + K * deltaU
    else:
        y_hat[idx] = y0 + K * deltaU * (1.0 - np.exp(-(t[idx] - theta)/tau))
    return y_hat

def plot_aberta_e_eqm(resultados):
    """
    Usa (t, y) e os parâmetros (K, tau, theta, deltaU, y0) para:
    1) gerar a curva prevista FOPDT;
    2) plotar Dados vs FOPDT;
    3) calcular EQM e RMSE e imprimir.
    """
    t      = resultados["t"]
    y      = resultados["y"]
    K      = resultados["K"]
    tau    = resultados["tau"]
    theta  = resultados["theta"]
    deltaU = resultados["deltaU"]
    y0     = resultados["y0"]

    # 1) curva prevista do modelo
    y_hat = fopdt_step_response(t, K, tau, theta, deltaU, y0)

    # 2) EQM (MSE) e RMSE
    eqm  = float(np.mean((y - y_hat)**2))
    rmse = float(np.sqrt(eqm))
    print(f"EQM  (MSE) = {eqm:.6f}")
    print(f"RMSE       = {rmse:.6f}")

    # 3) plot comparativo
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, y,      label="Dados (y)")
    plt.plot(t, y_hat,  "--", label=f"FOPDT (K={K:.3f}, τ={tau:.2f}s, θ={theta:.2f}s)\nRMSE={rmse:.3f}")
    plt.title("Malha aberta: Dados vs. FOPDT (Smith)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Saída y(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"EQM": eqm, "RMSE": rmse}


import math

def metodo_CHR(K, tau, theta, overshoot="0%"):
    """
    Sintonia CHR (servo) para FOPDT: Gp(s) = K * e^{-theta s} / (tau s + 1)
    Parâmetros:
      - K: ganho do processo
      - tau: constante de tempo do processo (s)
      - theta: atraso/tempo morto do processo (s)
      - overshoot: "0%" ou "20%"

    Retorna: (Kp, Ti, Td) no formato PID paralelo (u = Kp*(e + 1/Ti * ∫e + Td * de/dt))
    """
    if overshoot == "0%":
      # Sem overshoot:
      # Kp = 0.6 * (tau / (K * theta))
      # Ti = tau
      # Td = theta / 2
      Kp = 0.6 * (tau / (K * theta))
      Ti = tau
      Td = theta / 2.0

    elif overshoot == "20%":
      # Com ~20% de overshoot:
      # Kp = 0.95 * (tau / (K * theta))
      # Ti = 1.357 * tau
      # Td = 0.473 * theta
      Kp = 0.95 * (tau / (K * theta))
      Ti = 1.357 * tau
      Td = 0.473 * theta

    else:
      raise ValueError("overshoot deve ser '0%' ou '20%'")

    return Kp, Ti, Td


def metodo_Cohen_Coon(K, tau, theta):
    """
    Sintonia Cohen–Coon (curva de reação) para FOPDT: Gp(s) = K * e^{-theta s} / (tau s + 1)
    Fórmulas clássicas em função de R = theta/tau:

      Kp = (1/K) * (tau/theta) * (4/3 + R/4)
      Ti = tau * (32 + 6R) / (13 + 8R)
      Td = (4 * theta) / (11 + 2R)

    Retorna: (Kp, Ti, Td) no formato PID paralelo.
    """
    R = theta / tau  # razão atraso/constante de tempo

    Kp = (1.0 / K) * (tau / theta) * (4.0/3.0 + R/4.0)
    Ti = tau * (32.0 + 6.0*R) / (13.0 + 8.0*R)
    Td = (4.0 * theta) / (11.0 + 2.0*R)

    return Kp, Ti, Td

def simular_rastreio_setpoint(resultados, Kp, Ti, Td, SP, pade_ord=1, titulo="PID (Padé)"):
    """
    Simula malha fechada (PID paralelo + FOPDT com atraso via Padé) rastreando um setpoint SP.
    - resultados: dict da sua extrair_dados() (tem K, tau, theta, y0, y, t)
    - Kp, Ti, Td: ganhos do PID (paralelo)
    - SP: setpoint absoluto (mesma unidade de y)
    - pade_ord: ordem do Padé (1 ou 2)
    """
    # 1) pega infos do seu dataset/modelo
    t    = resultados["t"]
    ydat = resultados["y"]
    K    = float(resultados["K"])
    tau  = float(resultados["tau"])
    th   = float(resultados["theta"])
    y0   = float(resultados["y0"])

    # 2) eixo de tempo para simulação (começando em 0)
    T = t - t[0]

    # 3) planta FOPDT com Padé: Gp(s) = K/(tau s + 1) * e^{-th s}
    s = ctl.TransferFunction.s
    G_nom   = K / (tau*s + 1)
    num_d, den_d = ctl.pade(th, pade_ord)  # aproxima e^{-th s}
    G_delay = ctl.tf(num_d, den_d)
    Gp = G_nom * G_delay

    # 4) PID paralelo: Gc(s) = Kp * (1 + 1/(Ti s) + Td s)
    Gc = Kp
    if Ti not in (None, 0): Gc += Kp / (Ti * s)
    if Td not in (None, 0): Gc += Kp * Td * s

    # 5) malha fechada (rastreio): T(s) = Gc*Gp / (1 + Gc*Gp)
    Tcl = ctl.feedback(Gc * Gp, 1)

    # 6) resposta ao degrau **unitário** em variável de desvio (0 -> 1)
    tout, y_dev = ctl.step_response(Tcl, T=T)

    # 7) converter para **unidade absoluta** usando seu setpoint:
    #    var. de desvio = (y - y0) / (SP - y0)   =>  y = y0 + (SP - y0)*y_dev
    y_sim = y0 + (SP - y0) * y_dev

    # 8) gráfico (dataset vs simulação) + linha do setpoint
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, ydat, label="Dados (malha aberta)")        # só referência visual
    plt.plot(t, y_sim, "--", label=f"{titulo} (fechada, SP={SP})")
    plt.axhline(SP, linestyle=":", label="Setpoint (SP)")
    plt.title("Rastreamento de setpoint com PID (atraso via Padé)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Saída y(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return tout, y_sim

# --- exemplo de uso (na sua main) ---
if __name__ == "__main__":
    resultados = extrair_dados()  # sua função
    K, tau, theta = resultados["K"], resultados["tau"], resultados["theta"]

    # escolha do metodo (exemplos):
    #Kp, Ti, Td = metodo_CHR(K, tau, theta, overshoot="0%")          # CHR 0%
    # Kp, Ti, Td = metodo_CHR(K, tau, theta, overshoot="20%")       # CHR 20%
    Kp, Ti, Td = metodo_Cohen_Coon(K, tau, theta)                 # Cohen–Coon

    SP = 40             # <- este viria da sua interface (campo de texto)
    pade_ord = 1          # ou 2, se quiser refinar o atraso

    simular_rastreio_setpoint(resultados, Kp, Ti, Td, SP, pade_ord, titulo="CHR 0%")

    print(f"Kp = {Kp:.6f}, Ti = {Ti:.6f}, Td = {Td:.6f}")








