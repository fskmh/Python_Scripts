#!/usr/bin/env python
# Based on the code by Christo Venter in Matlab (King.m)
# Author : Bertie Seyffert
#
# 13-07-2026 mh
# Modified to work with newer versions of Numpy and Scipy where the array.ptp()
# method was deprecated and the function simps was replaced with simpson.

import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erf
from scipy.integrate import simpson  # Corrected import

from astropy import constants as const
G = const.G.cgs.value
pc = const.pc.cgs.value
M_sun = const.M_sun.cgs.value

# ==== ==== ==== ==== Arguments ==== ==== ==== ====
parser = argparse.ArgumentParser()

parser.add_argument('W_0', type=np.float64, help='Scaled potential at r = 0')
parser.add_argument('-d', '--data', default=None, type=str, help='Filename for data profile')
parser.add_argument('--dcR',  default=0, type=int, help='Column number for R')
parser.add_argument('--dcS',  default=1, type=int, help='Column number for sigma')
parser.add_argument('-o', '--output', default=None, type=str, help='Output filename')
parser.add_argument('-r', '--reference', default=None, type=str, help='Reference filename')
parser.add_argument('--dR', default=0.01, type=np.float64, help='Step length')
parser.add_argument('--v_disp', default=6.4e5, type=np.float64, help='Velocity dispersion')
parser.add_argument('--distance', default=8.5e3, type=np.float64, help='Distance to cluster')
parser.add_argument('--r_core', default=0.18, type=np.float64, help='Core radius')
parser.add_argument('--rho_0_infered', default=1e3, type=np.float64, help='Infered core density')
parser.add_argument('-m', '--mass', action='store_true', help='Output mass')
parser.add_argument('-s', '--suppress', action='store_true', help='Suppress graphs')

args = parser.parse_args()

# ---- Convert to cgs ----
args.distance *= pc
args.r_core = np.deg2rad(args.r_core/60.)*args.distance
args.rho_0_infered *= M_sun/np.power(pc, 3)

# ==== ==== ==== ==== Set up calculation ==== ==== ==== ====
def calcRho_over_k(W):
    return (
        np.pi*np.power(np.sqrt(2)*args.v_disp, 3)
        *(np.sqrt(np.pi)*np.exp(W)*erf(np.sqrt(W)) - 2*np.sqrt(W)*(W*2/3. + 1))
    )

def calcW(R_i, W_im1, W_i, rho_0, rho_i):
    F_0 = 4*np.pi*G*args.rho_0_infered*np.square(args.r_core/args.v_disp)
    return ((2*W_i - (1 - args.dR/R_i)*W_im1 - F_0*args.dR**2*rho_i/rho_0) / (1. + args.dR/R_i))

# ---- Initialise variables ----
R = np.array([0.])
W = np.array([args.W_0])
rho_over_k = np.array([calcRho_over_k(W=W[0])])

R = np.append(R, R[0]+args.dR)
W = np.append(W, W[0])
rho_over_k = np.append(rho_over_k, rho_over_k[0])

# ==== ==== ==== ==== Calculate sigma(R) ==== ==== ==== ====
W = np.append(W, calcW(R_i=R[-1], W_im1=W[-2], W_i=W[-1], rho_0=rho_over_k[0], rho_i=rho_over_k[-1]))

while W[-1] > 0.0:
    rho_over_k = np.append(rho_over_k, calcRho_over_k(W=W[-1]))
    R = np.append(R, R[-1] + args.dR)
    W = np.append(W, calcW(R_i=R[-1], W_im1=W[-2], W_i=W[-1], rho_0=rho_over_k[0], rho_i=rho_over_k[-1]))

W = np.delete(W, W.size-1)
print('Tidal radius: {0} pc'.format(R[-1]*args.r_core/pc))

# ---- Integrate over R to obtain sigma(R) ----
sigma = np.empty(R.size-1)
for index, r in enumerate(R[:-1]):
    # Changed simps to simpson
    sigma[index] = 2*simpson(
        y=(R[index+1:]/np.sqrt(np.square(R[index+1:]) - np.square(R[index]))) * rho_over_k[index + 1:],
        x=R[index+1:]
    )
sigma /= sigma.max()

# ==== ==== ==== ==== Derive Mass ==== ==== ==== ====
R *= args.r_core

if args.mass:
    rho = (args.rho_0_infered/rho_over_k[0])*rho_over_k
    # Changed simps to simpson
    M_total = 4.*np.pi*simpson(y=np.square(R)*rho, x=R)/M_sun
    print('Total mass of cluster: {0} M_sun'.format(M_total))

# ==== ==== ==== ==== Scale to data and save profile ==== ==== ==== ====
if args.data is not None:
    dataR, dataSigma = np.loadtxt(args.data, usecols=(args.dcR, args.dcS), unpack=True)
    # Changed array.ptp() method to np.ptp(array) function
    sigma = sigma*np.ptp(dataSigma) + dataSigma.min()

R /= pc

if args.output is not None:
     np.savetxt(args.output, np.column_stack((R[:-1], sigma)), fmt='%1.4g')

# ==== ==== ==== ==== Plot profiles ==== ==== ==== ====
if args.reference is not None:
    refR, refSigma = np.loadtxt(args.reference, unpack=True)
    refSigma -= refSigma.min()
    refSigma /= refSigma.max()
    # Changed array.ptp() method to np.ptp(array) function
    refSigma = refSigma*np.ptp(sigma) + sigma.min()

if not args.suppress:
    plt.plot(R[:-1], sigma/sigma.max(), label='Model profile')
    xmax = R.max()

    if args.reference is not None:
        plt.plot(refR, refSigma/refSigma.max(), color='green', label='Model profile (reference)')
        xmax = np.maximum(xmax, refR.max())

    if args.data is not None:
        plt.plot(dataR, dataSigma/dataSigma.max(), 'o', label='Observed profile')
        xmax = np.maximum(xmax, dataR.max())

    plt.title('Surface brightness vs. Radius')
    plt.xlabel('R (pc)')
    plt.ylabel('Surface brightness (normalised)')
    plt.xlim(0, 1.1*xmax)
    plt.ylim(0, 1.1)
    plt.legend(loc=1)
    plt.show()
