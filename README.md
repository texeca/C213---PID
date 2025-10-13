# Projeto de Controle PID

Este projeto implementa uma **Interface Gráfica do Usuário (GUI)** em Python (Tkinter) para a identificação de sistemas de Primeira Ordem com Tempo Morto (**FOPDT**) e a sintonia de controladores PID utilizando métodos clássicos (**CHR** e **Cohen-Coon**).

A aplicação permite a comparação direta entre as estratégias de sintonia, usando aproximação de Padé para simular o atraso de tempo morto.

-----

## Algoritmos de Controle Implementados

### 1\. Identificação do Modelo (Smith - Dois Pontos)

O modelo FOPDT é obtido a partir da curva de reação (resposta ao degrau) carregada do arquivo `.mat`.

| Parâmetro | Método | Característica |
| :--- | :--- | :--- |
| **FOPDT** | **Smith (Método dos Dois Pontos)** | Estima o Ganho ($K$), a Constante de Tempo ($\tau$) e o Tempo Morto ($\theta$) utilizando os tempos em $28.3\%$ e $63.2\%$ da variação total da saída ($\Delta y$). |
| **Atraso ($\theta$)** | **Aproximação de Padé** | O atraso é modelado com a **Aproximação de Padé** (ordem N=1 ou N=2) para permitir a simulação em malha fechada. |

### 2\. Sintonia do Controlador PID (Malha Fechada)

O controlador PID é sintonizado usando os parâmetros $K$, $\tau$ e $\theta$ obtidos na etapa de identificação.

#### a) Método de Cohen-Coon (CC)

Foco em respostas rápidas e são geralmente mais agressivos.

$$K_p = \frac{1}{K} \frac{\tau}{\theta} \left( \frac{4}{3} + \frac{1}{4} \frac{\theta}{\tau} \right)$$$$T_i = \tau \frac{32 + 6(\theta/\tau)}{13 + 8(\theta/\tau)}$$$$T_d = \theta \frac{4}{11 + 2(\theta/\tau)}$$

#### b) Método CHR (Chien, Hrones e Reswick)

Fornece configurações separadas para otimização de **controle regulatório** (rejeição de distúrbios) e **controle servo** (rastreamento de *setpoint*).

| Configuração | $\mathbf{K_p}$ | $\mathbf{T_i}$ | $\mathbf{T_d}$ |
| :--- | :--- | :--- | :--- |
| **0% Overshoot** (Regulatório) | $0.6 \frac{\tau}{K \theta}$ | $\tau$ | $0.5 \theta$ |
| **20% Overshoot** (Servo) | $0.95 \frac{\tau}{K \theta}$ | $1.357 \tau$ | $0.473 \theta$ |

-----

## Como Executar o Projeto

O arquivo principal do projeto se chama `main_GUI.py` e está localizado dentro de uma pasta chamada `codigos`.

### 1\. Pré-requisitos e Instalação de Dependências

Certifique-se de ter o Python instalado (versão 3.x recomendada). As dependências necessárias estão listadas no arquivo `requirements.txt`.

Abra o terminal na pasta raiz do projeto e execute:

```bash
# Instalar todas as bibliotecas necessárias
pip install -r requirements.txt
```

### 2\. Execução da Interface Gráfica

Após a instalação das dependências, execute o arquivo principal usando o interpretador Python:

```bash
python codigos/main_GUI.py
```

### 3\. Utilização

1.  **Aba Identificação:** Clique em **"Escolher .mat"** para carregar os dados de malha aberta. Em seguida, clique em **"Identificar (Smith)"**. O FOPDT será ajustado e plotado no gráfico.
2.  **Aba Controle PID:**
      * Insira o valor do **Setpoint (SP)** desejado.
      * Selecione um método individual (CHR ou CC) ou, para comparar, selecione **"Comparação (Todos)"**.
      * Clique em **"Simular (método)"**. O gráfico de rastreamento será atualizado, mostrando a resposta da saída $\mathbf{y(t)}$ para o(s) método(s) selecionado(s), e a tabela de métricas será exibida ao lado.
      * Caso deseje manualmente aplicar os valores, adicione os valores de Kp, Ti e Td desejados e clique em ***Simular com ganhos manuais***. O gráfico será atualizado, mostrando a simulação com os ganhos manuais.
