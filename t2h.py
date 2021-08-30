import numpy as np
from scipy.stats import f

def T2Hotelling(df, mu0, n, p):
    Xbarra=df.mean()
    S = df.cov()
    S_inv = np.linalg.inv(S)
    T2Hotelling = n*np.array(Xbarra-mu0).T.dot(S_inv).dot(np.array(Xbarra-mu0))
    qf = f.ppf(0.95, p , n-p, loc=0, scale=1)
    teste = T2Hotelling > (n-1) * p / (n-p) * qf
    pvalor = 1-f.cdf(T2Hotelling/((n-1) * p / (n-p) ), p, n-p)
    print('Rejeitamos H0') if teste else print('Não rejeitamos H0')
    print('Valor da estatística', T2Hotelling)
    print('valor p', pvalor)