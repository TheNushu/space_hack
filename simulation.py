import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import astropy.constants as ap_const
import pyIGRF


class aalto3:
    def __init__(self):

        self.m = 2  # kg

        # center of simulation is center of earth
        self.pos = np.array([0., 0., 0.])
        self.vel = np.array([0., 0., 0.])
        self.acc = np.array([0., 0., 0.])

        self.dim = np.array([0.1, 0.1, 0.1])  # m
        # assume homegonous dist. of mass
        self.I_middle = 1/12*self.m*(2*self.dim[0]**2)
        # self.I_edge =
        # self.I_diagon =
        self.inertia_tens = np.zeros((3, 3))
        for i in range(3):
            self.inertia_tens[i][i] = self.I_middle

# np.cross()


def dist_from_earth(vec):
    r = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)


def orbit_time_sec(r):
    # r += ap_const.R_earth.value
    t = np.sqrt(4*np.pi**2*r**3/(ap_const.G.value*ap_const.M_earth.value))
    return t


print(orbit_time_sec(500000)/60)

# def
sat = aalto3()
print(sat.inertia_tens)
