import numpy as np



def retorna_config(y_eq, r, q, n, a=1.0, b=2.0):
    np.set_printoptions(precision=16, suppress=False)
    alpha = y_eq(a)
    beta = y_eq(b)

    h = (b - a) / float(n)
    m = n - 1
    
    x = np.array([a + i * h for i in range(n + 1)], dtype=np.float64)

    A_h = np.zeros((m, m), dtype=np.float64)
    v_h = np.zeros(m,dtype=np.float64)

    # Preenchimento da Matriz A_h
    for i in range(m):
        A_h[i, i] = 2.0 + (h**2) * q(x[i + 1])
        if i > 0:
            A_h[i, i - 1] = -1.0
        if i < m - 1:
            A_h[i, i + 1] = -1.0

    # Preenchimenyo Vetor V_h
    for i in range(m):
        v_h[i] = - (h**2) * r(x[i + 1])
    v_h[0] += alpha
    v_h[m - 1] += beta

    return A_h, v_h, x, h



# --- Fatoração de Cholesky ---
def cholesky (A,y,n):
    np.set_printoptions(precision=16, suppress=False)

    # Encontrar U triangular superior tal que U^t * U = A
    U = np.zeros((n, n))

    U[0, 0] = np.sqrt(A[0, 0])

    for k in range(n):
        soma_k = sum(U[j, k]**2 for j in range(k))
        U[k, k] = np.sqrt(A[k, k] - soma_k)
        
        if k < n:
            for i in range(k + 1, n):
                soma_i = sum(U[j, k] * U[j, i] for j in range(k))
                U[k, i] = (1.0 / U[k, k]) * (A[i, k] - soma_i)

    # Resolver U^t * w = y (Substituição para frente)
    y_inter = np.zeros(n)
    for i in range(n):
        y_inter[i] = y[i]
        for j in range(i):
            y_inter[i]=y_inter[i]-U[j,i]*y_inter[j]
        y_inter[i]=y_inter[i]/U[i,i]


    # Resolver U * w = y_inter (Substituição para trás)
    x = np.zeros(n)#ver se no escopo geral permanece a var??
    for i in range(n - 1, -1, -1):

        x[i] = y_inter[i]
        for j in range(i+1,n):
            x[i]=x[i]-U[i,j]*x[j]
        x[i]=x[i]/U[i,i]

    return U,y_inter,x

def Times(A_vec,v):

    w = np.zeros(len(v), dtype=np.float64)
    
    for l in range(A_vec.shape[1]):
        i = int(A_vec[0, l])-1
        j = int(A_vec[1, l])-1
        
        w[i] = w[i] + (A_vec[2, l]) * v[j]
        
    return w

def to_Sparse(n,q,x, h): #Pela modularização preciso de h aqui
    M = np.zeros((3, (3*n-5)))

    M[0, 0], M[1, 0], M[2, 0] = 1, 1, (2 + (h**2) * q(x[0]))

    for j in range(n-1):
        # Diagonal principal
        M[0, j] = j + 1
        M[1, j] = j + 1
        M[2, j] = 2.0 + (h**2) * q(x[j + 1])
        
        # Super-diagonal
        M[0, n - 2 + j] = j
        M[1, n - 2 + j] = j + 1
        M[2, n - 2 + j] = -1.0
        
        # Sub-diagonal
        M[0, 2 * n - 4 + j] = j + 1
        M[1, 2 * n - 4 + j] = j
        M[2, 2 * n - 4 + j] = -1.0
    return M

def grad_conj(A, y, n, x_0, tol=1e-14, n_max=100, imprimir=False):
    from main import formatar_limpo

    r_0=(Times(A,x_0)-y)
    p_0=-r_0

    eps = tol+1
    k=0
    x_old=x_0.copy()
    x_new=x_0.copy()
    r_old=(Times(A,x_0)-y)
    r_new=r_old.copy()
    p_old=-r_old.copy()

    relatorio_str = ""
    while (eps>tol and k<n_max):

        if imprimir:
            relatorio_str += f"\nRelatório detalhado no cálculo da iteração {k+1}:\n"
            relatorio_str += f"x^{{(old)}}:\n{formatar_limpo(x_old)}\n"
            relatorio_str += f"r^{{(old)}}:\n{formatar_limpo(r_old)}\n"
            relatorio_str += f"p^{{(old)}}:\n{formatar_limpo(p_old)}\n"

        A_p = Times(A, p_old)
        alpha = np.dot(r_old, r_old) / np.dot(p_old, A_p)

        if imprimir:
            print(f"\nRelatório detalhado no cálculo da iteração {k+1}:")
            print("x^{(old)}:\n", x_old)
            print("r^{(old)}:\n", r_old)
            print("p^{(old)}:\n", p_old)
            print("alpha:\n", alpha)

        x_new = x_old + alpha * p_old
        
        r_new = Times(A, x_new) - y
        
        beta = np.dot(r_new, r_new) / np.dot(r_old, r_old)
        
        p_new = -r_new + beta * p_old
        
        if imprimir:
            print("x^{(new)}:\n", x_new)
            print("r^{(new)}:\n", r_new)
            print("beta:\n", beta)
            print("p^{(new)}:\n", p_new)

        # --- BLOCO NEW ---
        if imprimir:
            relatorio_str += f"x^{{(new)}}:\n{formatar_limpo(x_new)}\n"
            relatorio_str += f"r^{{(new)}}:\n{formatar_limpo(r_new)}\n"
            relatorio_str += f"beta:\n{beta:.15f}\n"
            relatorio_str += f"p^{{(new)}}:\n{formatar_limpo(p_new)}\n"
            relatorio_str += "-"*50 + "\n"

        eps = max(abs(a) for a in r_new)
        
        x_old = x_new.copy()
        r_old = r_new.copy()
        p_old = p_new.copy()

        k+=1
    return x_new, eps,k, relatorio_str