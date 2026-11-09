import numpy as np
from bib.metodos import to_Sparse, grad_conj, Times, retorna_config

# --- DEFINIÇÕES ---
def y_eq(x): return (x**2) * np.cosh(x)
def q(x): return 1.0
def r(x): return 2.0 * np.cosh(x) + 4.0 * x * np.sinh(x)

def executar_tarefa5(y_eq, q, r, n, a=1.0, b=2.0):

    _, v_h, x, h=retorna_config(y_eq, r, q, n, a, b)
    
    A_vec = to_Sparse(n, q, x, h)
    
    print("--- (A_h) vec ---")
    print(A_vec)
    
    x0 = np.zeros(n-1, dtype=np.float64)
    
    print("\n--- GRADIENTE CONJUGADO ---")
    x_sol, erro, k_total, relatorio_cg = grad_conj(A_vec, v_h, n, x0, n_max=3, imprimir=True)
    return A_vec, x_sol, k_total, erro, relatorio_cg

if __name__ == "__main__":
    def y_eq(x):
        return (x**2) * (np.e**x)

    def q(x):
        return 1.0+(4/x)

    def r(x):
        return 2.0 * np.exp(x)
    executar_tarefa5(y_eq, q, r, 5, 0.0,1.0)