import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def initial_conditions1():
    # u_t = a * u_xx + f(x, t) | equation
    # u(x, 0) = phi(x) | initial condition
    # alpha[0] * u(0, t) + beta[0] * u_x(0, t) = gamma[0](t) | left border condition
    # alpha[1] * u(1, t) + beta[1] * u_x(1, t) = gamma[1](t) | right border condition
    a = 1.
    sigma = 0.5
    phi = lambda x: -x/2
    f = lambda x, t: np.exp(x) / 2 * (1 / np.cos(t) / np.cos(t) - np.tan(t))
    alpha = np.array([1, 1])
    beta = np.array([1, 0])
    gamma = np.array([lambda t: np.tan(t) - 0.5, lambda t: (np.e * np.tan(t) - 1) / 2])
    u0 = lambda x, t: (np.exp(x)*np.tan(t) - x) / 2 # exact solution
    return a, sigma, phi, f, alpha, beta, gamma, u0


def initial_conditions2():
    # u_t = a * u_xx + f(x, t) | equation
    # u(x, 0) = phi(x) | initial condition
    # alpha[0] * u(0, t) + beta[0] * u_x(0, t) = gamma[0](t) | left border condition
    # alpha[1] * u(1, t) + beta[1] * u_x(1, t) = gamma[1](t) | right border condition
    a = 1.
    sigma = 0.5
    phi = lambda x: 0
    f = lambda x, t: np.exp(x) / 2 * (1 / np.cos(t) / np.cos(t) - np.tan(t))
    alpha = np.array([1, 1])
    beta = np.array([1, 0])
    gamma = np.array([lambda t: np.tan(t) - 0.5, lambda t: (np.e * np.tan(t) - 1) / 2])
    u0 = lambda x, t: 0
    return a, sigma, phi, f, alpha, beta, gamma, u0

def least_squares(x, y):
    n = len(x)

    sumx = x.sum()
    sumy = y.sum()
    xy = x * y
    sumxy = xy.sum()
    xx = x * x
    sumxx = xx.sum()

    b = (n * sumxy - sumx*sumy) / (n * sumxx - sumx**2)
    a = (sumy - b * sumx) / n
    return a, b


def solve3diagonal(a, b, c, f):
    # assuming that len(a) == len(b) == len(c) == len(f) == n + 1
    A = np.zeros(len(f))
    B = np.zeros(len(f))
    A[0] = -c[0] / b[0]
    B[0] = f[0] / b[0]
    for i in range(1, len(f) - 1):
        A[i] = -c[i] / (b[i] + a[i] * A[i - 1])
        B[i] = (f[i] - a[i] * B[i - 1]) / (b[i] + a[i] * A[i - 1])
    A[-1] = 0
    B[-1] = (f[-1] - a[-1] * B[-2]) / (b[-1] + a[-1] * A[-2])

    y = np.zeros(len(f))
    y[-1] = B[-1]
    for i in range(len(f) - 2, -1, -1):
        y[i] = B[i] + A[i] * y[i + 1]
    return y


def next_layer_first_order(u_prev, a2, sigma, func, alpha, beta, gamma, t_now, tau, h):
    a = np.zeros(len(u_prev))
    b = np.zeros(len(u_prev))
    c = np.zeros(len(u_prev))
    f = np.zeros(len(u_prev))
    coef = tau * a2 / h / h
    for i in range(1, len(u_prev) - 1):
        a[i] = coef * sigma
        b[i] = -1 - 2 * coef * sigma
        c[i] = coef * sigma
        f[i] = -u_prev[i] + coef * (sigma - 1) * (u_prev[i + 1] - 2 * u_prev[i] + u_prev[i - 1]) - tau*func(i * h, t_now - tau/2)
    a[0] = 0.
    b[0] = -alpha[0] / h + beta[0]
    c[0] = alpha[0] / h
    f[0] = gamma[0](t_now)
    a[-1] = -alpha[1] / h
    b[-1] = alpha[1] / h + beta[1]
    c[-1] = 0.
    f[-1] = gamma[1](t_now)
    u_next = solve3diagonal(a, b, c, f)
    return u_next


def next_layer_second_order(u_prev, a, sigma, phi, f, alpha, beta, gamma, t_now):
    return


def animation():
    x_min = 0.  # left border
    x_max = 1.  # right border
    t_min = 0.  # initial time
    t_max = 40.  # end time
    h = 0.01  # space step
    tau = h  # time step
    N = int((x_max - x_min) // h)  # number of points
    x_range = np.linspace(x_min, x_max, N)

    a, sigma, phi, f, alpha, beta, gamma, u0 = initial_conditions1()
    next_layer = next_layer_first_order

    u = np.zeros((2, N))
    u[0] = phi(x_range)

    fig = plt.figure()
    ax = plt.axes(xlim=(x_min, x_max), ylim=(-2, 2))
    line, = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        t_now = t_min + i * tau
        print(t_now)
        if i == 0:
            y = u[0]
        elif i == 1:
            u[1] = next_layer(u[0], a, sigma, f, alpha, beta, gamma, t_now, tau, h)
            y = u[1]
        else:
            u[0] = u[1]
            u[1] = next_layer(u[0], a, sigma, f, alpha, beta, gamma, t_now, tau, h)
            y = u[1]
        # y = y - u0(x_range, t_now)  # if there is need to plot the error
        line.set_data(x_range, y)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=int((t_max - t_min) // tau), blit=True)

    plt.show()
    anim.save('solution.gif', writer='imagemagick')


if __name__ == '__main__':
    animation()
