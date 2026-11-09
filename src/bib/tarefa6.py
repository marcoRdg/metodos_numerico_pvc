import numpy as np
from bib.metodos import to_Sparse, Times, grad_conj, retorna_config

# --- DEFINIÇÕES ---
def y_eq(x): return (x**2) * np.cosh(x)
def q(x): return 1.0
def r(x): return 2.0 * np.cosh(x) + 4.0 * x * np.sinh(x)

def executar_tarefa6(y_eq, q, r, n, a=1.0, b=2.0):

    _, v_h, x, h=retorna_config(y_eq, r, q, n, a, b)

    A_vec = to_Sparse(n, q, x, h)
    
    x0 = np.zeros(n-1, dtype=np.float64)
    
    x_final, eps_final, k,_ = grad_conj(A_vec, v_h, n, x0, tol=0.35, n_max=n-1, imprimir=False)
    
    return x_final, eps_final, k

if __name__ == "__main__":

    def y_eq(x):
        return (x**2) * (np.e**x)

    def q(x):
        return 1.0+(4/x)

    def r(x):
        return 2.0 * np.exp(x)
    x_final, eps, k = executar_tarefa6(y_eq, q, r, 10,0.0,1.0)
    
    print(f"--- SOLUÇÃO FINAL APÓS {k} ITERAÇÕES (tol = 0.35) ---")

    print(x_final)
    print("\nErro residual (eps):", eps)