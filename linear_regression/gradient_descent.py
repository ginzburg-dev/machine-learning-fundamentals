import math, copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import Slider

def f_wb_function(x, w, b):
    f_wb = w*x+b
    return f_wb

def cost_function(x, y, w, b):
    m = len(x)
    cost = 0
    for i in range(m):
        f_wb = w*x[i]+b
        cost = cost + (f_wb - y[i])**2
    cost = cost / (2*m)
    return cost

def calculate_gradient(x, y, w, b):
    m = len(x)
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w*x[i]+b
        dj_dw_i = (f_wb - y[i])*x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(x, y, w_init, b_init, alpha, iterations, cost_function, gradient_function):
    w = w_init
    b = b_init
    p_history = []
    J_history = []
    for i in range(iterations):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        cost = cost_function(x, y, w, b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        if np.isnan(cost) or np.isinf(cost):
            print("Divergence detected, stopping early.")
            break
        J_history.append(cost)
        p_history.append([w, b])
        if i % 100 == 0:
            print(f"Cost: {J_history[-1]:.2f} ",
                f"dj_dw: {dj_dw:.2f} ", f"dj_db: {dj_db:.2f} "
                f"W: {w:.2f} ", f"B: {b:.2f} ")
    
    return w, b, J_history, p_history

def draw_line(x0, y0, w, size):
    b = y0 - w*x0
    y1 = w*(x0+size) + b
    y2 = w*(x0-size) + b
    x1 = (x0+size)
    x2 = (x0-size)
    ax.plot([x1, x2], [y1, y2], linestyle='dashed')

x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([200.0, 350.0, 250.0, 320.0, 290.0])

# Calculate gradient descent
alpha = 1.0e-2
iterations = 100000
w_result, b_result, J_hist, p_hist = gradient_descent(x_train,y_train,0,0,alpha,iterations,cost_function,calculate_gradient)
p_hist_w = []
p_hist_b = []
J_hist_out = []
for i in range(len(J_hist)):
    p_hist_w.append(p_hist[i][0])
    p_hist_b.append(p_hist[i][1])
    J_hist_out.append(J_hist[i])

# Plot3D cost func with respect to w, and b
border = 50
g_range = np.max( [np.max([p[0] for p in p_hist]), np.max([p[1] for p in p_hist])] ) + border
w_val = np.linspace(-g_range, g_range)
b_val = np.linspace(-g_range, g_range)
W, B = np.meshgrid(w_val,b_val)
Z = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = cost_function(x_train, y_train, W[i, j], B[i, j])

fig, axs = plt.subplots(subplot_kw=dict(projection='3d'))

# Plot gradien descent 3D history
axs.plot3D(p_hist_w, p_hist_b, J_hist_out, c='magenta', linewidth=2, label='Gradient descent path')
surf = axs.plot_surface(W, B, Z, rstride=5, cstride=5, cmap='winter', linewidth=1, alpha=0.3, antialiased=False, shade=False, label='Cost function')

axs.set_title('3D cost function')
axs.set_xlabel('W (weight)')
axs.set_ylabel('B (bias)')
axs.set_zlabel('Cost')
axs.legend()

# Plot interactive cost graph
b_0 = 300
g_range_f2 = 50
w_vals_f2 = np.linspace(-g_range_f2, g_range_f2)
cost_vals = np.zeros_like(w_vals_f2)

for idx, w in enumerate(w_vals_f2):
    cost_vals[idx] = cost_function(x_train, y_train, w, b_0)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

ax.plot(w_vals_f2, cost_vals, marker='', c='b', label = "Prediction")
dashed_line, = ax.plot([], [], linestyle='dashed', color='r', label='Gradient Line')
plt.title("Cost function with respect to w. b=300")
plt.ylabel('Cost')
plt.xlabel('W')
plt.legend()

axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='W (weight)',
    valmin=-g_range_f2,
    valmax=g_range_f2,
    valinit=20,
)


# Update function
def update(val):
    w = freq_slider.val
    y = cost_function(x_train, y_train, w, b_0)
    slope = calculate_gradient(x_train, y_train, w, b_0)[0]
    size = 10
    b_line = y - slope * w
    x1 = w - size
    x2 = w + size
    y1 = slope * x1 + b_line
    y2 = slope * x2 + b_line

    for text in ax.texts:
        text.remove()

    dashed_line.set_data([x1, x2], [y1, y2])
    ax.text(w+5, y, r"$\frac{\partial J(w, b)}{\partial w}$ = " + str(round(slope)), fontsize=12)
    fig.canvas.draw_idle()

update(0)
freq_slider.on_changed(update)

# Plot cost vs. iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 

fig3, ax3 = plt.subplots()
result_line = []
for i in range(len(x_train)):
    result_line.append( f_wb_function(i, w_result,b_result) )
ax3.scatter(x_train, y_train, c='r', label='Real data')
ax3.plot(x_train, result_line, c='b', label='Solved linear regression')
ax3.legend()
plt.show()
