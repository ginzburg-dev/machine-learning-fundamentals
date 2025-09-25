import sympy

J, w = sympy.symbols('J, w')

J = w**3
print(J) # prints w**3

dJ_dw = sympy.diff(J, w)
print(dJ_dw) # prints 3*w**2

value = dJ_dw.subs(w, 2)
print(value) # prints 12

value = dJ_dw.subs([(w, 2)])
print(value) # prints 12


# another way to calculate derivatives
w = 3
epsilon = 0.0001
J = w**2
J_epsilon = (w + epsilon)**2 
dj_dw = (J_epsilon - J)/epsilon # 6.000100000012054
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= {dj_dw:0.6f} ")
