"""
File: duct_pso.py
Author: Cade Parkison
Email: cadeparkison@gmail.com
Github: c-park

Description: Particle-Swarm Optimization of duct design

"""

import sys

import pso
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def plot_cost():
    x = np.linspace(0.1, 0.7)
    y = np.linspace(0.25, 1)

    X,Y = np.meshgrid(x,y)


    c = np.dstack((X,Y))

    out = np.zeros(X.shape)

    for i, x in enumerate(c):
        for j, y in enumerate(x):
            out[i][j] = cost(np.array([y[0], y[1]])[np.newaxis])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    Z = out
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    #ax.set_zlim(3000, 5000)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.draw()

def diam(dp, L, Vfr, props):
    # Computes Diameter of a duct in m based on:

    g = props[0]        # gravity (m/s^2)
    rho = props[1]      # Density of air (kg/m^3)
    eps = props[2]      # Absolute roughness of galvanized duct with seems
    nu = props[3]       # Kinematic viscosity

    hl = dp/(rho*g)

    D = 0.66*(eps**1.25*((L*Vfr**2/g)*(hl)**(-1))**4.75 + nu*Vfr**9.4*(
                (L/g)*hl**(-1))**5.2)**0.04
    return D


def p_loss(D, L, Q, props):
    """computes pressure loss in duct

    Args:
        :D: (scalar) Diameter of duct (meters)
        :L: (scalar) Length of duct (meters)
        :Q: (scalar) Volumetric flow rate (m^3/s)
        :props: List of properties

    Returns:
        :dp: (scalar) Pressure loss (N/m^2)

    """
    g = props[0]
    rho = props[1]
    eps = props[2]
    nu = props[3]

    # head loss (m)
    h1 = 1.07*Q**2*L*(np.log(eps*(3.7*D)**(-1) + 4.62*(nu*D/Q)**0.9))**(-2)*(
            g*D**5)**-1
    dp = rho*g*h1

    return dp


def cost(X):
    """Computes total cost of the duct system

    :X: (numpy array)
            :rows: particle
            :columns: guessed dimension

    :returns: Total System Cost

    """

    # Known Sectional Properties
    #     H = height
    #     D = Diameter
    #     L = length
    #     Q = volumetric flow rates
    #     C = loss coefficients
    #     dpf = additional pressure loss
    H1 = 0.254
    D3 = 0.33
    L1, L2, L3, L4, L5 = 14, 12, 8, 16, 19.81
    Q1, Q2, Q3, Q4, Q5 = 0.7, 0.22, 0.92, 0.5, 1.42
    C1, C2, C3, C4, C5 = 0.8, 0.65, 0.18, 0.65, 1.5
    dpf1, dpf2, dpf5 = 25, 37.5, 37.5

    # System Properties
    g = 9.81          # Gravity (m/s^2)
    rho = 1.2         # Air Density (kg/m^3)
    eps = 3*10**-4    # Absolute roughness of galvanized duct
    nu = 1.54*10**-5  # kinematic viscosity
    props = [ g, rho, eps, nu ]

    # Cost Properties
    Cfan = 0      # Cost of fan
    Ec = 0.0203   # unit energy cost ($/kWh)
    T = 4400      # run time ( hr/year)
    Sd = 43.27    # unit ductwork cost ($/m^2)
    pwef = 8.61   # present worth escalation factor
    nf = 0.75     # fan operating efficiency
    ne = 0.8      # motor efficiency

    n = 1
    d = 2

    H1, W1, D2, D3, D4, D5, Pfan = dimensions(X)

    # compute Cost
    Qfan = Q5
    Ca = Qfan*Ec*T/(nf*ne*1000)*Pfan        # annual operating cost
    Co = Ca*pwef                            # operating cost in 10 years
    Ci = ((2*H1 + 2*W1)*L1 + L2*np.pi*D2 + L3*np.pi*D3 + L4*np.pi*D4
          + L5*np.pi*D5)*Sd
    Ci = Ci + np.ones((n))*Cfan
    C = Ci + Co

    return C

def dimensions(X):
    """Computes total cost of the duct system

    :X: (numpy array)
            :rows: particle
            :columns: guessed dimension

    :returns: (scalar) Total System Cost

    """

    # Known Sectional Properties
    #     H = height
    #     D = Diameter
    #     L = length
    #     Q = volumetric flow rates
    #     C = loss coefficients
    #     dpf = additional pressure loss
    H1 = 0.254
    D3 = 0.33
    L1, L2, L3, L4, L5 = 14, 12, 8, 16, 19.81
    Q1, Q2, Q3, Q4, Q5 = 0.7, 0.22, 0.92, 0.5, 1.42
    C1, C2, C3, C4, C5 = 0.8, 0.65, 0.18, 0.65, 1.5
    dpf1, dpf2, dpf5 = 25, 37.5, 37.5

    # System Properties
    g = 9.81          # Gravity (m/s^2)
    rho = 1.2         # Air Density (kg/m^3)
    eps = 3*10**-4    # Absolute roughness of galvanized duct
    nu = 1.54*10**-5  # kinematic viscosity
    props = [ g, rho, eps, nu ]

    #print(X.shape)
    n, d = X.shape
    W1 = X[:,0].T
    H1 = np.ones((n))*H1
    D5 = X[:,1].T
    #print('W1 = {}'.format(W1.shape))
    #print('H1 = {}'.format(H1.shape))
    #print('D5 = {}'.format(D5.shape))

    # compute dp1
    D1 = (1.3*(W1*H1)**0.625)*((W1 + H1)**(-0.25))
    #print('D1 = {}'.format(D1.shape))
    A1 = W1*H1
    v1 = Q1 / A1
    dp1 = p_loss(D1, L1, Q1, props) + dpf1 + C1*rho*(v1**2)/2

    # compute D2 from dp2
    dp2 = dp1
    dpc2 = np.zeros((n))
    dpf2 = np.ones((n))*dpf2
    D2old = np.zeros((n))
    error = 1
    nD2 = 0
    while error > 0.01:
        dp2d = dp2 - dpf2 - dpc2
        D2 = diam(dp2d, L2, Q2, props)
        #print('D2 = {}'.format(D2.shape))
        error = ((D2 - D2old).T.dot(D2 - D2old)/(D2.T.dot(D2)))**0.5
        D2old = D2
        A2 = np.pi*D2**2 / 4
        v2 = Q2/A2
        dpc2 = C2*rho*(v2**2) / 2
        nD2 = nD2 + 1

    # compute dp3 from D3
    dp3 = p_loss(D3, L3, Q3, props)
    A3 = np.pi*D3**2 / 4
    v3 = Q3 / A3
    dpc3 = C3 * rho * (v3**2) / 2
    dp3 = dp3 + dpc3

    # compute D4 from dp4
    dp4 = dp3 + dp1
    dpc4 = np.zeros((n))
    D4old = np.zeros((n))
    error = 1
    nD4 = 0
    while error > 0.01:
        dp4d = dp4 - dpc4
        D4 = diam(dp4d, L4, Q4, props)
        error = ((D4 - D4old).T.dot(D4 - D4old)/(D4.T.dot(D4)))**0.5
        D4old = D4
        A4 = np.pi*D4**2 / 4
        v4 = Q4/A4
        dpc4 = C4*rho*(v4**2) / 2
        nD4 = nD4 + 1

    # compute dp5 given D5
    A5 = np.pi*D5**2 / 4
    v5 = Q5 / A5
    dpc5 = C5*rho*(v5**2) / 2
    dp5 = p_loss(D5, L5, Q5, props) + np.ones((n))*dpf5 + dpc5

    Pfan = dp5 + dp4

    return H1, W1, D2, D3, D4, D5, Pfan

def pso_solve():
    """ Finds optimum diameters of duct system using pso algorithm

    :returns: TODO

    """
    plt.ion() # turn on interactive mode (otherwise the window arises not before "show()")
    fig = plt.figure();
    ax = fig.gca()
    #xs = np.array([])
    #ys = np.array([])
    ax.set_ylim(0.15, 1.1)
    ax.set_xlim(0,0.8)

    # plot empty line to generate line object
    #line, = ax.scatter(xs,ys)

    n = 2    # number of particles
    d = 2    # number of dimensions

    xmax = np.array([0.7, 1.0])
    xmin = np.array([0.1, 0.25])

    w = 0.7
    c1 = 2
    c2 = 2

    R = np.random.rand(n,d)
    #R = np.ones((n,d))
    x = np.ones((n,d)).dot(np.diag(xmin)) + R.dot(np.diag(xmax - xmin))
    R = np.random.rand(n,d)
    Delx = (R - 0.5).dot(np.diag((xmax - xmin)/2))

    F = cost(x)

    fopti = F
    xopti = x
    xoptg = x[0,:]
    foptg = F[0]

    fxmin, ixmin = np.min(F), np.argmin(F)

    if fxmin < foptg:
        foptg = fxmin
        xoptg = x[ixmin, :]

    max_delx = np.ones((n,d)).dot(np.diag((xmax-xmin)/4))

    for i in range(100):
        #Ri = np.array([[0.6,0.5],[0.4,.8]])
        #Rg = np.array([[0.2,0.7],[0.8,.3]])
        #Ri = np.ones((n,d))*0.5
        #Rg = np.ones((n,d))*0.5
        Ri = np.random.rand(n,d)
        Rg = np.random.rand(n,d)
        Delx = Delx*w + c1*Ri*(xopti - x) + c2*Rg*(np.ones((n,d)).dot(
                                                            np.diag(xoptg)) -x)
        x += Delx

        # limit particles to within boundary and limit deltaX
        for particle in x:
            for j in range(len(particle)):
                if particle[j] > xmax[j]:
                    particle[j] = xmax[j]
                    Delx[j] = -1*max_delx[j]
                if particle[j] < xmin[j]:
                    particle[j] = xmin[j]
                    Delx[j] = max_delx[j]

        F = cost(x)

        for j, f in enumerate(F):
            if f < fopti[j]:
                fopti[j] = f
                xopti[j] = x[j]
            if f <  foptg:
                foptg = f
                xoptg = x[j]

        #xs = np.append(xs, xoptg[0])
        #ys = np.append(ys, xoptg[1])
        #line.set_data(xs, ys)
        ax.scatter(xoptg[0], xoptg[1])
        plt.pause(.1)
        #plt.draw()

        print('i={:5} x=({:5.4}, {:5.4}) f={:5.4}'.format(i, xoptg[0], xoptg[1], foptg))

    return dimensions(xoptg[np.newaxis])

if __name__ == "__main__":
    dimensions = pso_solve()
    plt.ioff()
    print('Final dimensions\n')
    print('{:^9}{:^9}{:^9}{:^9}{:^9}{:^9}'.format('H1', 'W1', 'D2', 'D3', 'D4',
                                                  'D5'))
    print('---------------------------------------------------')
    print('{:^8.3} {:^8.3} {:^8.3} {:^8.3} {:^8.3} {:^8.3}'.format(dimensions[0][0],
                                               dimensions[1][0],
                                               dimensions[2][0],
                                               dimensions[3],
                                               dimensions[4][0],
                                               dimensions[5][0]))
