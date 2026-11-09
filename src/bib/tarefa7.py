import numpy as np
import time
from bib.metodos import to_Sparse, grad_conj, Times, cholesky, retorna_config

# --- DEFINIÇÕES ---
def y_eq(x): return (x**2) * np.cosh(x)
def q(x): return 1.0
def r(x): return 2.0 * np.cosh(x) + 4.0 * x * np.sinh(x)

def executar_tarefa7(y_eq, q, r, a=1.0, b=2.0):

    np.set_printoptions(precision=20, suppress=False, linewidth=120)  
    
    valores_n = list(range(100, 1001, 100))
    
    erros_chow, erros_cg, razao_erros = [], [], []
    tempos_chow, tempos_cg, razao_tempos = [], [], []

    for n in valores_n:

        A_h, v_h, x, h=retorna_config(y_eq, r, q, n, a, b)

        A_vec = to_Sparse(n, q, x, h)
        
        proj_y = np.array([y_eq(x[i + 1]) for i in range(n-1)])
        
        # CHOLESKY

        inicio_chow = time.perf_counter()
        _, _, w_chow = cholesky(A_h, v_h, n-1)
        fim_chow = time.perf_counter()
        
        tempo_chow = fim_chow - inicio_chow
        erro_chow = np.max(np.abs(proj_y - w_chow))

        # GRADIENTES CONJUGADOS

        x0 = np.zeros(n-1, dtype=np.float64)
        tol_cg = 1.0 / (n**2)
        n_max_cg = n-1
        
        inicio_cg = time.perf_counter()
        w_cg, _, _,_ = grad_conj(A_vec, v_h, n-1, x0, tol=tol_cg, n_max=n_max_cg, imprimir=False)
        fim_cg = time.perf_counter()
        
        tempo_cg = fim_cg - inicio_cg
        erro_cg = np.max(np.abs(proj_y - w_cg))
        

        # RESULTADOS

        erros_chow.append(erro_chow)
        erros_cg.append(erro_cg)
        razao_erros.append(erro_cg / erro_chow)
        
        tempos_chow.append(tempo_chow)
        tempos_cg.append(tempo_cg)
        razao_tempos.append(tempo_cg / tempo_chow)
        
    return valores_n, erros_chow, erros_cg, razao_erros, tempos_chow, tempos_cg, razao_tempos

if __name__ == "__main__":
    def y_eq(x):
        return (x**2) * (np.e**x)

    def q(x):
        return 1.0+(4/x)

    def r(x):
        return 2.0 * np.exp(x)
    n_vals, e_chow, e_cg, r_erro, t_chow, t_cg, r_tempo = executar_tarefa7(y_eq, q, r, 0.0,1.0)
    
    # Imprime os arrays
    np.set_printoptions(precision=7, suppress=True, linewidth=120)
    print("> erro CG / erro Cholewski")
    print(np.array(r_erro))
    
    np.set_printoptions(precision=4, suppress=True)
    print("\n> TimesChow")
    print(np.array(t_chow))
    print("> TimesCG")
    print(np.array(t_cg))
    
    np.set_printoptions(precision=6, suppress=True)
    print("\n> TimesCG / TimesChow")
    print(np.array(r_tempo))