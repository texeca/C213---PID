# -*- coding: utf-8 -*-
"""
Testes automatizados para main_GUI.py
Executa testes básicos sem abrir a interface gráfica.
"""

import numpy as np
import control as ctl
from main_GUI import extrair_dados, metodo_CHR, metodo_Cohen_Coon, simular_rastreio_setpoint

# ======== TESTE 1: Dados simulados e identificação FOPDT ========

def gerar_dados_sinteticos():
    """Gera um conjunto de dados sintéticos no formato esperado pelo extrair_dados()."""
    t = np.linspace(0, 100, 1000)
    K_true, tau_true, theta_true = 2.0, 15.0, 5.0
    u = np.ones_like(t) * 75.0  # degrau
    y = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] >= theta_true:
            y[i] = 50 + K_true * 75.0 * (1 - np.exp(-(t[i]-theta_true)/tau_true))
        else:
            y[i] = 50
    return {"t": t, "u": u, "y": y, "y0": 50, "y_inf": y[-1], "deltaU": 75.0}

def test_identificacao():
    """Teste simples de cálculo de FOPDT."""
    data = gerar_dados_sinteticos()
    t, u, y = data["t"], data["u"], data["y"]
    # Usa os parâmetros teóricos para simular e comparar
    K, tau, theta = 2.0, 15.0, 5.0
    from main_GUI import fopdt_step_response
    y_hat = fopdt_step_response(t, K, tau, theta, 75.0, 50)
    rmse = np.sqrt(np.mean((y - y_hat)**2))
    print(f"[Teste 1] RMSE entre dados e modelo FOPDT: {rmse:.4f}")

# ======== TESTE 2: Métodos de Sintonia ========

def test_metodos_sintonia():
    """Verifica se os métodos de sintonia retornam valores válidos."""
    K, tau, theta = 2.0, 15.0, 5.0
    chr0 = metodo_CHR(K, tau, theta, "0%")
    chr20 = metodo_CHR(K, tau, theta, "20%")
    cc = metodo_Cohen_Coon(K, tau, theta)
    print(f"[Teste 2] CHR (0%): Kp={chr0[0]:.3f}, Ti={chr0[1]:.3f}, Td={chr0[2]:.3f}")
    print(f"[Teste 2] CHR (20%): Kp={chr20[0]:.3f}, Ti={chr20[1]:.3f}, Td={chr20[2]:.3f}")
    print(f"[Teste 2] Cohen–Coon: Kp={cc[0]:.3f}, Ti={cc[1]:.3f}, Td={cc[2]:.3f}")

# ======== TESTE 3: Simulação de rastreamento ========

def test_simulacao_pid():
    """Testa a resposta ao setpoint com um PID sintonia CHR."""
    resultados = gerar_dados_sinteticos()
    K, tau, theta = 2.0, 15.0, 5.0
    Kp, Ti, Td = metodo_CHR(K, tau, theta, "0%")
    SP = resultados["y_inf"] + 10
    t, y_sim = simular_rastreio_setpoint(resultados, Kp, Ti, Td, SP)
    print(f"[Teste 3] Saída final simulada: y(t_f) = {y_sim[-1]:.3f}")

# ======== EXECUTAR ========

if __name__ == "__main__":
    print("===== INICIANDO TESTES =====")
    test_identificacao()
    test_metodos_sintonia()
    test_simulacao_pid()
    print("===== TESTES FINALIZADOS =====")
