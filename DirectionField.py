import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

fig = plt.figure(num=1)
ax=fig.add_subplot(111)

#  The ODE to be solved
def ODE(x, y):  # (t, y)
    return 1 + y**2

# Constants
n = 100
n_small = 20
xmin = -np.pi/2
xmax = np.pi/2
ymin = -10
ymax = 10

t_span = np.array([-np.pi*29/60, np.pi*29/60])
shifts = [0, -np.pi/8, np.pi/8]
init_vals = [(np.tan(t_span[0]) + k,) for k in shifts]
t_evals = [np.linspace(t_span[0] + k, t_span[1] + k, n) for k in shifts]

for k in range(len(shifts)):
    sol = solve_ivp(ODE, t_span + shifts[k], init_vals[k], t_eval=t_evals[k])
    if sol.success:
        ax.plot(sol.t, sol.y[0])

#  Vectorveld van alleen de standaardoplossing
#  x = np.linspace(xmin, xmax, n_small)
#  y = np.tan(x)
#  xv = np.ones(n_small)
#  yv = 1 + y**2
#  norm = np.sqrt(xv**2 + yv**2)
#  xv, yv = xv/norm, yv/norm
#  ax.quiver(x, y, xv, yv, units='xy', angles='xy')

# Vectorveld op hele plot
x = np.linspace(xmin, xmax, n_small)
y = np.linspace(ymin, ymax, n_small)
xm, ym = np.meshgrid(x, y)
xv = np.ones((n_small, n_small))
yv = 1 + y**2
norm = np.sqrt(xv**2 + yv**2)
xv, yv = xv/norm, yv/norm
ax.quiver(xm, ym, xv, yv, units='xy', angles='xy')

ax.plot(np.pi / 4, 1, 'r.')

plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.show()
