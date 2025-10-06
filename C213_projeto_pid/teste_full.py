# -*- coding: utf-8 -*-
# GUI PID: Identificação (Smith) + Controle (CHR/CC) com atraso Padé
# Requisitos: numpy, matplotlib, control, scipy

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # backend para Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import control as ctl
from scipy.io import loadmat
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os

# ========== Núcleo (identificação, FOPDT, sintonias, simulação) ==========

def extrair_dados(path=None):
    """
    Carrega o .mat (Formato A: variáveis 'tiempo','entrada','salida'),
    estima K,tau,theta (Smith) e retorna dicionário com tudo que precisamos.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("Caminho do .mat inválido ou não selecionado.")

    data = loadmat(path)

    # Formato A
    try:
        t = np.asarray(data['tiempo']).ravel().astype(float)
        u = np.asarray(data['entrada']).ravel().astype(float)
        y = np.asarray(data['salida']).ravel().astype(float)
    except KeyError as e:
        raise KeyError(f"Não encontrei a variável {e} no .mat. Esperado: 'tiempo', 'entrada', 'salida'.")

    # Patamares (robustos)
    y0    = float(np.median(y[: max(5, len(y)//50) ]))
    y_inf = float(np.median(y[- max(20, len(y)//20) :]))
    dy = y_inf - y0

    # ΔU assumido (entrada foi logada já no patamar final)
    deltaU = 75.0
    K = dy / deltaU if deltaU != 0 else 0.0

    # Tempos 28.3% e 63.2% (metodo de Smith)
    y_283 = y0 + 0.283 * dy
    y_632 = y0 + 0.632 * dy
    t_283 = float(np.interp(y_283, y, t))
    t_632 = float(np.interp(y_632, y, t))

    tau   = 1.5 * (t_632 - t_283)
    theta = t_632 - tau

    return {
        "t": t, "u": u, "y": y,
        "y0": y0, "y_inf": y_inf, "dy": dy,
        "deltaU": deltaU, "K": K,
        "t_283": t_283, "t_632": t_632,
        "tau": tau, "theta": theta,
        "mat_path": path
    }

def fopdt_step_response(t, K, tau, theta, deltaU, y0):
    t = np.asarray(t, dtype=float)
    y_hat = np.full_like(t, fill_value=y0, dtype=float)
    idx = t >= theta
    if tau <= 0:
        y_hat[idx] = y0 + K * deltaU
    else:
        y_hat[idx] = y0 + K * deltaU * (1.0 - np.exp(-(t[idx] - theta)/tau))
    return y_hat

def plot_aberta_e_eqm_axes(ax, t, y, y_hat, K, tau, theta):
    ax.clear()
    ax.plot(t, y, label="Dados (y)")
    ax.plot(t, y_hat, "--", label=f"FOPDT (K={K:.3f}, τ={tau:.2f}, θ={theta:.2f})")
    ax.set_title("Malha aberta: Dados vs. FOPDT (Smith)")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Saída y(t)")
    ax.grid(True)
    ax.legend()

def metodo_CHR(K, tau, theta, overshoot="0%"):
    if overshoot == "0%":
        Kp = 0.6 * (tau / (K * theta))
        Ti = tau
        Td = theta / 2.0
    elif overshoot == "20%":
        Kp = 0.95 * (tau / (K * theta))
        Ti = 1.357 * tau
        Td = 0.473 * theta
    else:
        raise ValueError("overshoot deve ser '0%' ou '20%'")
    return Kp, Ti, Td

def metodo_Cohen_Coon(K, tau, theta):
    R = theta / tau
    Kp = (1.0 / K) * (tau / theta) * (4.0/3.0 + R/4.0)
    Ti = tau * (32.0 + 6.0*R) / (13.0 + 8.0*R)
    Td = (4.0 * theta) / (11.0 + 2.0*R)
    return Kp, Ti, Td

def simular_rastreio_setpoint(resultados, Kp, Ti, Td, SP, pade_ord=1):
    """ PID paralelo + planta FOPDT com atraso via Padé; retorna (t_out, y_sim). """
    t    = resultados["t"]
    y0   = float(resultados["y0"])
    K    = float(resultados["K"])
    tau  = float(resultados["tau"])
    th   = float(resultados["theta"])

    T = t - t[0]  # tempo relativo para simulação

    s = ctl.TransferFunction.s
    G_nom   = K / (tau*s + 1)
    num_d, den_d = ctl.pade(th, pade_ord)
    G_delay = ctl.tf(num_d, den_d)
    Gp = G_nom * G_delay

    Gc = Kp
    if Ti not in (None, 0): Gc += Kp / (Ti * s)
    if Td not in (None, 0): Gc += Kp * Td * s

    Tcl = ctl.feedback(Gc * Gp, 1)               # rastreamento r->y
    tout, y_dev = ctl.step_response(Tcl, T=T)     # resposta unitária (variável de desvio)
    y_sim = y0 + (SP - y0) * y_dev               # volta para unidade absoluta

    return t, y_sim

# ---- Métricas de malha fechada (sobre rastreamento ao SP) ----
def metricas_fechada(t, y_sim, SP, y0, banda=0.02):
    """
    Calcula: tempo de subida (10-90%), tempo de acomodação (banda%), erro estacionário (média final).
    """
    t = np.asarray(t, float)
    y = np.asarray(y_sim, float)
    step = SP - y0
    if step == 0:
        return {"tr_10_90": np.nan, "ts_2": np.nan, "ess": 0.0}

    # Normaliza em relação ao passo: 0 no início, 1 no SP
    y_rel = (y - y0) / step

    # Tempo de subida 10-90% (interpolação)
    try:
        t10 = float(np.interp(0.10, y_rel, t))
        t90 = float(np.interp(0.90, y_rel, t))
        tr = max(0.0, t90 - t10)
    except Exception:
        tr = np.nan

    # Tempo de acomodação (2%): primeiro instante a partir do qual fica dentro da banda e não sai mais
    tol = banda * abs(step)
    dentro = np.abs(y - SP) <= tol
    # varre de trás pra frente para achar o primeiro índice a partir do qual está sempre dentro
    valid = dentro[::-1].cumprod()[::-1].astype(bool)
    idx = np.argmax(valid)  # primeiro True em valid
    ts = float(t[idx]) if valid.any() else np.nan

    # Erro estacionário: média do erro nas últimas 5% amostras
    n_tail = max(5, len(y)//20)
    ess = float(np.mean(y[-n_tail:] - SP))

    return {"tr_10_90": tr, "ts_2": ts, "ess": ess}

# ========== GUI (Tkinter) ==========

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Projeto PID (Identificação + Controle)")
        self.geometry("1120x740")

        self.resultados = None          # dicionário retornado por extrair_dados
        self.selected_mat_path = None   # caminho do .mat escolhido

        nb = ttk.Notebook(self)
        self.tab_id = ttk.Frame(nb)
        self.tab_pid = ttk.Frame(nb)
        nb.add(self.tab_id, text="Identificação (Smith)")
        nb.add(self.tab_pid, text="Controle PID")
        nb.pack(fill="both", expand=True)

        self.build_tab_identificacao()
        self.build_tab_pid()

    # ---- Aba Identificação ----
    def build_tab_identificacao(self):
        frame_left = ttk.Frame(self.tab_id, padding=10)
        frame_right = ttk.Frame(self.tab_id, padding=10)
        frame_left.pack(side="left", fill="y")
        frame_right.pack(side="right", fill="both", expand=True)

        # Controles (esquerda)
        ttk.Label(frame_left, text="1) Escolher dataset (.mat)").pack(anchor="w")
        ttk.Button(frame_left, text="Escolher .mat", command=self.on_choose_mat).pack(fill="x", pady=(2,8))
        self.lbl_mat = ttk.Label(frame_left, text="(nenhum arquivo selecionado)", foreground="#555")
        self.lbl_mat.pack(anchor="w")

        ttk.Label(frame_left, text="2) Identificar (Smith)").pack(anchor="w", pady=(10,2))
        ttk.Button(frame_left, text="Identificar (Smith)", command=self.on_identificar).pack(fill="x")

        ttk.Separator(frame_left, orient="horizontal").pack(fill="x", pady=8)
        self.txt_id = tk.Text(frame_left, width=38, height=18)
        self.txt_id.pack(pady=4)

        # Gráficos (direita)
        fig1 = matplotlib.figure.Figure(figsize=(6.2, 3.6), dpi=100)
        self.ax_u = fig1.add_subplot(211)
        self.ax_y = fig1.add_subplot(212, sharex=self.ax_u)
        self.canvas1 = FigureCanvasTkAgg(fig1, master=frame_right)
        self.canvas1.get_tk_widget().pack(fill="both", expand=True)

        fig2 = matplotlib.figure.Figure(figsize=(6.2, 3.6), dpi=100)
        self.ax_fopdt = fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(fig2, master=frame_right)
        self.canvas2.get_tk_widget().pack(fill="both", expand=True)

    def on_choose_mat(self):
        path = filedialog.askopenfilename(
            title="Selecionar arquivo .mat",
            filetypes=[("MATLAB file", "*.mat"), ("Todos", "*.*")]
        )
        if path:
            self.selected_mat_path = path
            base = os.path.basename(path)
            self.lbl_mat.config(text=f"Selecionado: {base}", foreground="#0a0")
        else:
            self.lbl_mat.config(text="(nenhum arquivo selecionado)", foreground="#555")

    def on_identificar(self):
        if not self.selected_mat_path:
            messagebox.showwarning("Atenção", "Escolha um arquivo .mat antes de identificar.")
            return
        try:
            self.resultados = extrair_dados(self.selected_mat_path)
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao identificar: {e}")
            return

        t = self.resultados["t"]; u = self.resultados["u"]; y = self.resultados["y"]
        K = self.resultados["K"]; tau = self.resultados["tau"]; theta = self.resultados["theta"]
        y0 = self.resultados["y0"]; y_inf = self.resultados["y_inf"]; deltaU = self.resultados["deltaU"]

        # Gráfico u(t) e y(t)
        self.ax_u.clear(); self.ax_y.clear()
        self.ax_u.plot(t, u); self.ax_u.set_title("Entrada u(t)"); self.ax_u.set_ylabel("u"); self.ax_u.grid(True)
        self.ax_y.plot(t, y); self.ax_y.set_title("Saída y(t)"); self.ax_y.set_xlabel("Tempo (s)"); self.ax_y.set_ylabel("y"); self.ax_y.grid(True)
        self.canvas1.draw()

        # FOPDT + EQM
        y_hat = fopdt_step_response(t, K, tau, theta, deltaU, y0)
        eqm = float(np.mean((y - y_hat)**2)); rmse = float(np.sqrt(eqm))
        plot_aberta_e_eqm_axes(self.ax_fopdt, t, y, y_hat, K, tau, theta)
        self.canvas2.draw()

        # Texto
        self.txt_id.delete("1.0", tk.END)
        self.txt_id.insert(tk.END, f"Arquivo: {os.path.basename(self.selected_mat_path)}\n\n")
        self.txt_id.insert(tk.END, f"K = {K:.6f}\nτ = {tau:.6f}\nθ = {theta:.6f}\n")
        self.txt_id.insert(tk.END, f"y0 = {y0:.4f}\ny_inf = {y_inf:.4f}\n")
        self.txt_id.insert(tk.END, f"EQM = {eqm:.6f}\nRMSE = {rmse:.6f}\n")

    # ---- Aba PID ----
    def build_tab_pid(self):
        frame_left = ttk.Frame(self.tab_pid, padding=10)
        frame_right = ttk.Frame(self.tab_pid, padding=10)
        frame_left.pack(side="left", fill="y")
        frame_right.pack(side="right", fill="both", expand=True)

        # Controles de método
        ttk.Label(frame_left, text="Método de Sintonia").pack(anchor="w")
        self.metodo_var = tk.StringVar(value="CHR0")
        ttk.Radiobutton(frame_left, text="CHR (0% OS)",   variable=self.metodo_var, value="CHR0").pack(anchor="w")
        ttk.Radiobutton(frame_left, text="CHR (20% OS)",  variable=self.metodo_var, value="CHR20").pack(anchor="w")
        ttk.Radiobutton(frame_left, text="Cohen–Coon",    variable=self.metodo_var, value="CC").pack(anchor="w")

        ttk.Separator(frame_left, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(frame_left, text="Setpoint (SP):").pack(anchor="w")
        self.sp_var = tk.StringVar(value="")
        ttk.Entry(frame_left, textvariable=self.sp_var).pack(fill="x")

        ttk.Label(frame_left, text="Padé (ordem)").pack(anchor="w", pady=(6,0))
        self.pade_var = tk.IntVar(value=1)
        ttk.Radiobutton(frame_left, text="1", variable=self.pade_var, value=1).pack(anchor="w")
        ttk.Radiobutton(frame_left, text="2", variable=self.pade_var, value=2).pack(anchor="w")

        ttk.Separator(frame_left, orient="horizontal").pack(fill="x", pady=6)
        # Botões: simular (método) e simular manual
        btns_frame = ttk.Frame(frame_left)
        btns_frame.pack(fill="x")
        ttk.Button(btns_frame, text="Simular (método)", command=self.on_simular).pack(side="left", expand=True, fill="x", padx=(0,4))
        ttk.Button(btns_frame, text="Salvar gráfico", command=self.on_salvar_grafico).pack(side="left", expand=True, fill="x")

        # Ganhos manuais
        ttk.Label(frame_left, text="Ganhos manuais (PID paralelo)").pack(anchor="w", pady=(10,0))
        manual_frame = ttk.Frame(frame_left)
        manual_frame.pack(fill="x")
        ttk.Label(manual_frame, text="Kp").grid(row=0, column=0, sticky="w")
        ttk.Label(manual_frame, text="Ti").grid(row=1, column=0, sticky="w")
        ttk.Label(manual_frame, text="Td").grid(row=2, column=0, sticky="w")
        self.kp_var = tk.StringVar(value="")
        self.ti_var = tk.StringVar(value="")
        self.td_var = tk.StringVar(value="")
        ttk.Entry(manual_frame, textvariable=self.kp_var, width=12).grid(row=0, column=1, padx=4, pady=2)
        ttk.Entry(manual_frame, textvariable=self.ti_var, width=12).grid(row=1, column=1, padx=4, pady=2)
        ttk.Entry(manual_frame, textvariable=self.td_var, width=12).grid(row=2, column=1, padx=4, pady=2)
        ttk.Button(frame_left, text="Simular com ganhos manuais", command=self.on_simular_manuais).pack(fill="x", pady=(6,0))

        self.txt_pid = tk.Text(frame_left, width=40, height=18)
        self.txt_pid.pack(pady=8)

        # Gráfico de rastreamento (direita)
        fig = matplotlib.figure.Figure(figsize=(6.8, 5.2), dpi=100)
        self.ax_track = fig.add_subplot(111)
        self.canvas_pid = FigureCanvasTkAgg(fig, master=frame_right)
        self.canvas_pid.get_tk_widget().pack(fill="both", expand=True)

    def _pegar_sp(self):
        try:
            sp_text = self.sp_var.get().strip()
            if sp_text == "":
                return float(self.resultados["y_inf"])
            return float(sp_text)
        except Exception:
            raise ValueError("Setpoint inválido.")

    def _simular_e_plotar(self, Kp, Ti, Td, SP, titulo, pade_ord):
        # Simula
        t, y_sim = simular_rastreio_setpoint(self.resultados, Kp, Ti, Td, SP, pade_ord=pade_ord)

        # Plot dataset vs rastreamento
        ydat = self.resultados["y"]
        self.ax_track.clear()
        self.ax_track.plot(self.resultados["t"], ydat, label="Dados (aberta)")
        self.ax_track.plot(self.resultados["t"], y_sim, "--",
                           label=f"{titulo} (fechada, SP={SP:.2f}, Padé N={pade_ord})")
        self.ax_track.axhline(SP, linestyle=":", label="Setpoint")
        self.ax_track.set_title("Rastreamento de setpoint com PID")
        self.ax_track.set_xlabel("Tempo (s)")
        self.ax_track.set_ylabel("Saída y(t)")
        self.ax_track.grid(True)
        self.ax_track.legend()
        self.canvas_pid.draw()

        # Métricas: overshoot, RMSE vs SP, tempo de subida (10–90%), tempo de acomodação (2%), erro estacionário
        rmse_sp = float(np.sqrt(np.mean((y_sim - SP)**2)))
        passo = SP - float(self.resultados["y0"])
        Mp = max(0.0, (np.max(y_sim) - SP) / abs(passo) * 100.0) if passo != 0 else 0.0
        mets = metricas_fechada(self.resultados["t"], y_sim, SP, self.resultados["y0"], banda=0.02)

        self.txt_pid.delete("1.0", tk.END)
        self.txt_pid.insert(tk.END, f"Método/Ganhos: {titulo}\nPadé: N={pade_ord}\n\n")
        self.txt_pid.insert(tk.END, f"Kp = {Kp:.6f}\nTi = {Ti:.6f}\nTd = {Td:.6f}\n\n")
        self.txt_pid.insert(tk.END, f"Overshoot ≈ {Mp:.1f}%\nRMSE (vs SP) ≈ {rmse_sp:.4f}\n")
        self.txt_pid.insert(tk.END, f"t_subida (10–90%) ≈ {mets['tr_10_90']:.3f} s\n")
        self.txt_pid.insert(tk.END, f"t_acomodação (2%) ≈ {mets['ts_2']:.3f} s\n")
        self.txt_pid.insert(tk.END, f"erro estacionário (média final) ≈ {mets['ess']:.4f}\n")

    def on_simular(self):
        if self.resultados is None:
            messagebox.showwarning("Atenção", "Primeiro faça a Identificação (aba anterior).")
            return
        try:
            SP = self._pegar_sp()
        except ValueError as e:
            messagebox.showerror("Erro", str(e))
            return

        K = self.resultados["K"]; tau = self.resultados["tau"]; theta = self.resultados["theta"]
        metodo = self.metodo_var.get()
        pade_ord = int(self.pade_var.get())

        if metodo == "CHR0":
            Kp, Ti, Td = metodo_CHR(K, tau, theta, overshoot="0%")
            titulo = "CHR (0% OS)"
        elif metodo == "CHR20":
            Kp, Ti, Td = metodo_CHR(K, tau, theta, overshoot="20%")
            titulo = "CHR (20% OS)"
        else:
            Kp, Ti, Td = metodo_Cohen_Coon(K, tau, theta)
            titulo = "Cohen–Coon"

        try:
            self._simular_e_plotar(Kp, Ti, Td, SP, titulo, pade_ord)
        except Exception as e:
            messagebox.showerror("Erro na simulação", str(e))

    def on_simular_manuais(self):
        if self.resultados is None:
            messagebox.showwarning("Atenção", "Primeiro faça a Identificação (aba anterior).")
            return
        try:
            SP = self._pegar_sp()
        except ValueError as e:
            messagebox.showerror("Erro", str(e))
            return

        try:
            Kp = float(self.kp_var.get().strip())
            Ti = float(self.ti_var.get().strip()) if self.ti_var.get().strip() != "" else 0.0
            Td = float(self.td_var.get().strip()) if self.td_var.get().strip() != "" else 0.0
        except ValueError:
            messagebox.showerror("Erro", "Ganhos manuais inválidos. Preencha Kp, Ti e Td como números.")
            return

        pade_ord = int(self.pade_var.get())
        try:
            self._simular_e_plotar(Kp, Ti, Td, SP, titulo="Ganhos manuais", pade_ord=pade_ord)
        except Exception as e:
            messagebox.showerror("Erro na simulação", str(e))

    def on_salvar_grafico(self):
        # Salva o gráfico atual da aba PID (rastreamento)
        if self.canvas_pid is None:
            return
        path = filedialog.asksaveasfilename(
            title="Salvar gráfico",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if not path:
            return
        try:
            self.canvas_pid.figure.savefig(path, bbox_inches="tight", dpi=150)
            messagebox.showinfo("OK", f"Gráfico salvo em:\n{path}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao salvar: {e}")

# ========== run ==========
if __name__ == "__main__":
    app = App()
    app.mainloop()
