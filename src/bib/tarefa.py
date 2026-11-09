import numpy as np

from bib.metodos import cholesky, retorna_config

def y_eq(x):
    return (x**2) * (np.e**x)

def q(x):
    return 1.0+(4/x)

def r(x):
    return 2.0 * np.exp(x)



def resolver_para_n(y_eq, r, q, n, a=1.0, b=2.0):
    A_h, v_h, x, h = retorna_config(y_eq, r, q, n, a, b)


    # Solução por Cholesky
    U, y_inter,w = cholesky(A_h,v_h,n-1)
    # Cálculo do Erro E_n
    proj_y = np.array([y_eq(x[i + 1]) for i in range(n-1)])
    vetor_erro = proj_y - w
    E_n = np.max(np.abs(vetor_erro))

    #np.set_printoptions(precision=30, suppress=False)

    print("--- MATRIZ A_h ---")
    print(A_h)
    print("\n--- VETOR v_h ---")
    print(v_h)
    print("\n--- MATRIZ U (Cholesky) ---")
    print(U)
    print("\n--- VETOR INTERMEDIÁRIO y ---")
    print(y_inter)
    print("\n--- SOLUÇÃO w_h ---")
    print(w)
    print("\n--- VETOR ERRO ---")
    print(vetor_erro)
    print("\n--- ERRO E_n (Norma Infinito) ---")
    print(E_n)

    return E_n, A_h, v_h, U, y_inter, w, vetor_erro

def executar_tarefa(y_eq, q, r, a=1.0, b=2.0):
    valores_n = list(range(100, 1001, 100))
    erros_En = []
    erros_normalizados = []

    for n in valores_n:
        E_n, _,_,_,_,_,_ = resolver_para_n(y_eq, r, q, n, a, b)
        erros_En.append(E_n)
        
        erro_norm = E_n * (n**2)
        erros_normalizados.append(erro_norm)

    print(f"{'n':>6} | {'E_n':>22} | {'E_n / (1/n^2)':>22}")
    print("-" * 56)
    for n, e, en in zip(valores_n, erros_En, erros_normalizados):
        print(f"{n:6d} | {e:22.15e} | {en:22.15f}")
    return valores_n, erros_En, erros_normalizados

if __name__ == "__main__":

    def y_eq(x):
        return (x**2) * (np.e**x)

    def q(x):
        return 1.0+(4/x)

    def r(x):
        return 2.0 * np.exp(x)

    #resolver_para_n(y_eq, q, r, 5, 0.0, 1.0)
    executar_tarefa(y_eq, q, r, 0.0,1.0)