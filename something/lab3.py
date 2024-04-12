#Метод множителей Лагранжа, метод Зойтендейка, метод проекции градиента Розена

import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize_scalar
import sympy as sm 
from sympy import symbols 


x, y, l = symbols('x y l')
f =  x - y
g =  1.5*x**2 + y**2 - 1
lg = f + l*g
grad = np.array([sm.diff(lg, x), sm.diff(lg, y), sm.diff(lg, l)])
print(sm.solve(grad, (x, y, l)))


class ZTDK:

    def __init__(self, dim, f, constraints, grad):
        self.dim = dim
        self.f = f
        self.constraints = constraints
        self.grad = grad

    def search_a_max(self, A2, b2, s, x):
        b_new = b2 - A2 @ x
        s_new = A2 @ s
        a_list = []
        for i in range(self.dim):
            if(s_new[i] > 0):
                a_list.append(b_new[i]/s_new[i])
        if(len(a_list) == 0):
            a_max = 99999
        else:
            a_max = np.min(a_list)
        return a_max

    def resolve_active_constraints(self, x):
        A1, A2, b1, b2 = [], [], [], []
        for constraint in self.constraints:
            if(abs(constraint(x)) < eps):
                A1.append(self.constraints[constraint][0])
                b1.append(0)
            else:
                A2.append(self.constraints[constraint][0])
                b2.append(self.constraints[constraint][1])

        return np.array(A1), np.array(A2), np.array(b1), np.array(b2)

    def optimize(self, x0, eps):
        x = x0
        A1, A2, b1, b2 = self.resolve_active_constraints(x)
        counter = 0
        while True:
            c = self.grad(x)
            s0_bounds = [-1, 1]
            s1_bounds = [-1, 1]

            res = linprog(c, A_ub=A1, b_ub=b1, bounds=[s0_bounds, s1_bounds])
            s = res.x

            if(abs(c.T @ s) < eps):
                break
            
            a_max = self.search_a_max(A2, b2, s, x)
            res = minimize_scalar(lambda alpha: self.f(x + alpha * s), bounds=(0, a_max), method='bounded')
            alpha = res.x
            x = x + s*alpha
            A1, A2, b1, b2 = self.resolve_active_constraints(x)

            counter += 1
            print(f'iteration : {counter} || current_position: {x}')

        return x


f = lambda x: - 10*x[1] - 2*x[0]*x[1] + 6 * x[0]**2 + x[1]**2
grad = lambda x: np.array([12*x[0] - 2*x[1], -2*x[0] + 2*x[1] - 10])

g1 = lambda x: 2*x[0] + x[1] - 5
g2 = lambda x: -2*x[0] - x[1] + 2
g3 = lambda x: -x[0]
g4 = lambda x: -x[1]

eps = 1e-2

d = {
    g1 : [[2, 1], [5]],
    g2 : [[-2, -1], [-2]],
    g3 : [[-1, 0], [0]],
    g4 : [[0, -1], [0]]
}

x0 = np.array([0, 4])

ztd = ZTDK(2, f, d, grad)
print(ztd.optimize(x0, eps))