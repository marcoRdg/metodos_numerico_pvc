import numpy as np
import time
from bib.metodos import to_Sparse, grad_conj, Times, cholesky, retorna_config

def executar_tarefa8(y_eq, q, r, a=1.0, b=2.0):
    # a, b = 1.0, 2.0
    # alpha_val = y_eq(a)
    # beta_val = y_eq(b)
    
    valores_n = [2000]
    
    tempos_chow, tempos_cg, razao_tempos = [], [], []
    
    for n in valores_n:
        print(f"-> Processando malha para n = {n}")

        A_h, v_h, x, h=retorna_config(y_eq, r, q, n, a, b)
        # h = (b - a) / n
        # m = n - 1
        
        # x = np.array([a + i * h for i in range(n + 1)], dtype=np.float64)
        
        # Matriz Densa (Cholesky)
        # A_h = np.zeros((m, m), dtype=np.float64)
        # for i in range(m):
        #     A_h[i, i] = 2.0 + (h**2) * q(x[i + 1])
        #     if i > 0: A_h[i, i - 1] = -1.0
        #     if i < m - 1: A_h[i, i + 1] = -1.0
            
        # Matriz Esparsa (CG)
        A_vec = to_Sparse(n, q, x, h)
        
        # # Vetor v_h
        # v_h = np.zeros(m, dtype=np.float64)
        # for i in range(m):
        #     v_h[i] = - (h**2) * r(x[i + 1])
        # v_h[0] += alpha_val
        # v_h[m - 1] += beta_val
        
        # EXECUÇÃO: CHOLESKY

        inicio_chow = time.perf_counter()
        cholesky(A_h, v_h, n-1)
        fim_chow = time.perf_counter()
        tempo_chow = fim_chow - inicio_chow
        
        # EXECUÇÃO: GRADIENTES CONJUGADOS

        x0 = np.zeros(n-1, dtype=np.float64)
        tol_cg = 1.0 / (n**2)
        n_max_cg = n-1
        
        inicio_cg = time.perf_counter()
        grad_conj(A_vec, v_h, n-1, x0, tol=tol_cg, n_max=n_max_cg, imprimir=False)
        fim_cg = time.perf_counter()
        tempo_cg = fim_cg - inicio_cg
        
        # Armazenar resultados
        tempos_chow.append(tempo_chow)
        tempos_cg.append(tempo_cg)
        razao_tempos.append(tempo_cg / tempo_chow)
        
        print(f"   Tempo Cholesky: {tempo_chow:.2f}s | Tempo CG: {tempo_cg:.2f}s")
        
    return valores_n, tempos_chow, tempos_cg, razao_tempos

if __name__ == "__main__":

    def y_eq(x):
        return (x**2) * (np.e**x)

    def q(x):
        return 1.0+(4/x)

    def r(x):
        return 2.0 * np.exp(x)
    
    n_vals, t_chow, t_cg, r_tempo = executar_tarefa8(y_eq, q, r, 0.0, 1.0)
    
    np.set_printoptions(precision=4, suppress=True)
    print("\n> TimesChow")
    print(np.array(t_chow))
    print("> TimesCG")
    print(np.array(t_cg))
    
    np.set_printoptions(precision=6, suppress=True)
    print("\n> TimesCG / TimesChow")
    print(np.array(r_tempo))