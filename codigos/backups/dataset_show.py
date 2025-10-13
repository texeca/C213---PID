from scipy.io import loadmat, whosmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


path = r"/datasets/Dataset_Grupo10_c213.mat"

# lista “o que tem lá dentro” (nome, tamanho, tipo)
print(whosmat(path))

# carrega o arquivo
data = loadmat(path)
print(type(data), len(data))      # dicionário: chave -> variável do MATLAB
print(data.keys())                # nomes das variáveis salvas


t = np.asarray(data['tiempo']).ravel().astype(float)
u = np.asarray(data['entrada']).ravel().astype(float)
y = np.asarray(data['salida']).ravel().astype(float)
dados_entrada = np.asarray(data['dados_entrada']).ravel().astype(float)
dados_salida = np.asarray(data['dados_saida']).ravel().astype(float)

print(t.shape, u.shape, y.shape)
print(t[:5], '...', t[-5:])  # preview
print(u[:5], '...', u[-5:])
print(y[:5], '...', y[-5:])

plt.figure(figsize=(8,4))
plt.plot(t, u, label='entrada u(t)')
plt.plot(t, y, label='saída y(t)')
plt.xlabel('tempo (s)')
plt.ylabel('valor')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

df = pd.DataFrame({
        "tempo [s]": t,
        "entrada u(t)": u,
        "saida y(t)": y,

        #"dados entrada(t)": dados_entrada,
        #"dados saida(t)": dados_salida
    })


print(df.head(30))   # mostra só as primeiras n linhas



