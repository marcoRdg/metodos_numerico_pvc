import numpy as np
from bib.metodos import Times

def executar_tarefa4():

    M_vec_test = np.array([
        [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
        [1, 3, 4, 2, 3, 2, 3, 4, 1, 2, 3, 4],
        [0.6, 1.0, 1.0, 2.0, -1.0, 1.0, 3.0, 1.0, 0.8, -2.0, -1.0, 1.0]
    ], dtype=np.float64)
    
    v_test = np.array([2.34, 1.17, -0.99, -0.78], dtype=np.float64)
    w_test = Times(M_vec_test, v_test)
    
    M_vec = np.array([
        [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 6],
        [1, 2, 3, 6, 3, 5, 2, 6, 5, 6, 3, 2, 4, 6],
        [-1.0, -1.0, 2.0, 1.0, 4.0, 3.0, 1.0, 2.0, 2.0, 2.0, 5.0, 7.0, 7.0, 7.0]
    ], dtype=np.float64)
    
    v = np.array([0.37, 0.14, -0.3, 0.8, 1.0, -1.2], dtype=np.float64)
    
    w = Times(M_vec, v)
    
    return w_test, w

if __name__ == "__main__":
    w_test, w = executar_tarefa4()
    
    np.set_printoptions(precision=16, suppress=False)

    print("\n(M_vev_test)")
    print(w_test)

    print("\n(M_vec * v)")
    print(w)