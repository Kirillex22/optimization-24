import math
import numpy as np
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")


def Gauss(x, func, n=2, eps = 0.05):
    y = x
    j = 1
    e = np.eye(n)
    while(True):
        f = lambda x: func(y+x*e[j-1])
        result = minimize_scalar(f)
        lmbd = result.x
        y = y + lmbd*e[j-1]
        if j == n:
            x_pred = x
            x = y
            if np.linalg.norm(x - x_pred)<eps:
                break
            else:
                j=1
        else:
            j+=1
    print("return")
    return x


def Method1(x, global_func, r, beta, eps = 0.05):
    while (True):
        result_minimze = Gauss(x, global_func)
        if abs(penalty_func1(result_minimze))<eps:
            break
        else:
            r*=beta

    return result_minimze


def Method2(x, global_func, r, beta, eps = 0.05):
    while (True):
        result_minimze = Gauss(x, global_func)
        if abs(penalty_func2(result_minimze))<eps:
            break
        else:
            r*=beta

    return result_minimze

#Func and df
func = lambda x: x[0]**2 + x[1]**2 + 4*x[1] - 1
dfdx = lambda x: -x[0]**2 - x[1]
dfdy = lambda x: -x[0] + 2*x[1] + 8

#Штрафы
limit1 = lambda x: x[0]-x[1]**2
limit2 = lambda x: -1*(x[0]+x[1]-2)
#Первый способо штрафа
limits1 = lambda x: 1/(limit1(x)) + 1/(limit2(x))
#Второй способ штрафа
limits2 = lambda x: -1*np.log(limit1(x)) + -1*np.log(limit2(x))

#Метод штрафных
penalty_func1 = lambda x: limits1(x)
global_func1 = lambda x: func(x) + r*penalty_func1(x)

#Метод барьеров
penalty_func2 = lambda x: (max(-(dfdx(x)), 0))**beta + (max(-(dfdy(x)), 0))**beta 
global_func2 = lambda x: func(x) + r*(penalty_func2(x))


x0 = [0.5, -1.5]
eps = 0.05
r = 0.01
beta = 2

result1 = Method1(x0, global_func1, r, beta, eps)
print(round(result1[0], 3), round(result1[1], 3))
result2 = Method2(x0, global_func2, r, beta, eps)
print(round(result2[0], 3), round(result2[1], 3))

