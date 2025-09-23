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
