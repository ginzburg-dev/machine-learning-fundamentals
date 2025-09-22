import math, copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import Slider

def plane3D(x, z, w1, w2, b):
    f_xz = w1*x + w2*z + b
    return f_xz

w1_init = 1
w2_init = -0.5
b_init = 5
x_train = np.linspace(-10, 10)
z_train = np.linspace(-10, 10)
X, Z = np.meshgrid(x_train, z_train)
Y = np.zeros_like(X)



# Plot 3D
fig, axs = plt.subplots(subplot_kw=dict(projection='3d'))
surf = axs.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap='winter', linewidth=1, alpha=0.3, antialiased=False, shade=False, label='Plane')

axs.set_title('3D plane')
axs.set_xlabel('X')
axs.set_ylabel('Z')
axs.set_zlabel('Y')
axs.legend()

ax_w1 = plt.axes([0.25, 0.15, 0.65, 0.03])
w1_slider = Slider(
    ax=ax_w1,
    label='w1',
    valmin=-100,
    valmax=100,
    valinit=0,
)

ax_w2 = plt.axes([0.25, 0.10, 0.65, 0.03])
w2_slider = Slider(
    ax=ax_w2,
    label='w2',
    valmin=-100,
    valmax=100,
    valinit=0,
)

ax_b = plt.axes([0.25, 0.05, 0.65, 0.03])
b_slider = Slider(
    ax=ax_b,
    label='b',
    valmin=-100,
    valmax=100,
    valinit=0,
)

def update(val):
    global surf
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            Y[i, j] = plane3D(X[i, j], Z[i, j], w1_slider.val, w2_slider.val, b_slider.val)
    surf.remove()
    surf = axs.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap='winter', linewidth=1, alpha=0.3, antialiased=False, shade=False, label='Cost function')
    fig.canvas.draw_idle()

update(0)
w1_slider.on_changed(update)
w2_slider.on_changed(update)
b_slider.on_changed(update)

plt.show()
