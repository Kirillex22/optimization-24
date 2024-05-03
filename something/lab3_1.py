# #Метод множителей Лагранжа, метод Зойтендейка, метод проекции градиента Розена

# import numpy as np
# from scipy.optimize import linprog
# from scipy.optimize import minimize_scalar
# from sympy import *  



# # x0 = np.array([0.5, 0.5])
# # x, y, l = sm.simbols('x y l')
# # f = x - y
# # g = 1.5*x**2 + y**2 - 1

# # #Lagrange
# # lg = f + l*g
# # gradient = lambda x, y, l: np.array([diff(lg, x), diff(lg, y), diff(lg, l)])
# # result = solve(gradient, (x, y, l))
# # point = np.array([result[x], result[y]])



# class ZTDK:

#     def __init__(self, dim, f, constraints, grad_f, grad_g):
#         self.dim = dim
#         self.f = f
#         self.constraints = constraints
#         self.grad_f = grad_f
#         self.grad_g = grad_g

#     def search_a_max(self, A2, b2, s, x):
#         if len(A2) == 0 or len(b2) == 0:
#             return 99999
#         b_new = b2 - A2 @ x
#         s_new = A2 @ s
#         a_list = []
#         for i in range(self.dim):
#             if(s_new[i] > 0):
#                 a_list.append(b_new[i]/s_new[i])
#         if(len(a_list) == 0):
#             a_max = 99999
#         else:
#             a_max = np.min(a_list)
#         return a_max

#     def resolve_active_constraints(self, x):
#         A1, A2, b1, b2 = [], [], [], []
#         for constraint in self.constraints:
#             if(abs(constraint(x)) < eps):
#                 A1.append(self.constraints[constraint][0])
#                 b1.append(0)
#             else:
#                 A2.append(self.constraints[constraint][0])
#                 b2.append(self.constraints[constraint][1])

#         return np.array(A1), np.array(A2), np.array(b1), np.array(b2)

#     def optimize(self, x0, eps):
#         x = x0
#         stop = False
#         while not stop:
#             c = self.grad(x)

#             s0_bounds = [-1, 1]
#             s1_bounds = [-1, 1]
#             res = linprog(c, A_ub=A1, b_ub=b1, bounds=[s0_bounds, s1_bounds])



# # f = lambda x: - x[0] - x[1]
# # grad = lambda x: np.array([1, -1])
# # g = lambda x: 1.5*x[0]**2 + x[1]**2 - 1
# # eps = 1e-2
# # d = {g : [[1.5, 1], [1]]}
# # x0 = np.array([0, 1])


# f = lambda x: - x[0] - x[1]
# grad_f = lambda x: np.array([1, -1])
# g = lambda x: 1.5*x[0]**2 + x[1]**2 - 1
# grad_g = lambda x: np.array([3*x[0], 2*x[1]])
# eps = 1e-2
# x0 = np.array([1, 1])

# ztd = ZTDK(2, f, d, grad)
# print(ztd.optimize(x0, eps))


def find_max(mas: list) -> int:
    max_i = -9999999999999999
    for i in mas:
        if i > max_i:
            max_i = i

    return max_i



mas = [1,31,3,13,1,3,43,54,566,5,7657,6,423,56,45,7]

print(f"real max{max(mas)}   our max{find_max(mas)}")
