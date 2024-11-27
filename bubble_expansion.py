"""
Functions for estimating the probability of spontaneous embolism due to bubble expansion. For details of the formalism, see Ingram, Reischl, Vesala & Vehkamäki 2024,
https://pubs.rsc.org/en/content/articlehtml/2024/na/d4na00316k

Code written by Stephen Ingram
"""

import numpy as np
from scipy.special import erf
from tqdm import tqdm


def nlip(r,apl):
    # function to calculate nlipids
    # from radius and area per lipid
    area = np.square(r)
    area *= 4*np.pi
    return np.divide(area,apl)


def r0(apl,nlipids):
    # function to calculate radius
    # from apl and nlipids
    area = apl*nlipids
    area /= 4*np.pi
    return np.sqrt(area)


def sigmoid(r,A,r0,std):
    arg = (r-r0+std)/std
    surf = erf(arg)
    surf *= (A/2)
    surf += (A/2)
    surf /= 1000
    return surf

def CNT_area(r,p,nlipids,APL,T):
    # Boltzmann constant
    kb = 1.38e-23  # J/K
    # Planck constant
    h = 6.62607e-34 # J s
    # Tolman length
    sig = 6.32e-11
    # Volume of bubble
    v = (4/3)*np.pi*np.power(r,3)
    # Young Laplace equation
    denom = 1+np.divide(2*sig,r)
    # get current sigmoid center radius in nm
    cent = r0(APL,nlipids)/1e9
    # calculate surface tension at current apl
    # note that this is a fit to MD data

    # It should be made non-hard coded at some point
    # 270 K
    # gamma = sigmoid(r,63,cent,0.62e-8)+sigmoid(r,14,cent+16e-9,0.9e-8)
    # 290 K
    # gamma = sigmoid(r,59,cent,0.66e-8) + sigmoid(r,15,cent+16e-9,0.9e-8)
    # 300 K
    gamma = sigmoid(r, 57, cent, 0.68e-8) + sigmoid(r, 15.5, cent+16e-9, 0.9e-8)
    # 310 K
    # gamma = sigmoid(r,55,cent,0.7e-8) + sigmoid(r,16,cent+16e-9,0.9e-8)

    # surface free energy per unit area
    surf = np.divide(gamma,denom)
    # surface free energy
    f = np.multiply(surf,4*np.pi*np.square(r))
    # volumetric work (pv) in m^3*Pa
    g = np.add(f,np.multiply(v,p))
    return np.divide(g,kb*T)


def probability(p, T, mu, sigma, apl, r_range, n_bubble):
    
    # lognormal size distribution of bubbles
    # in nm
    r_init = np.random.lognormal(mu, sigma, n_bubble)

    # start by assuming an equilibrium APL of 0.6
    # calculate the number of lipids for each bubble
    coatings = nlip(r_init,apl).astype(int)

    # matrix of whether it embolises or not
    emb = np.zeros(shape=(np.shape(r_init)))

    # convert from nm to m
    r_init /= 1e9
    assert np.max(r_init) <= np.max(r_range)

    for j in range(np.size(r_init)):
        gibbs = CNT_area(r_init[j],p,coatings[j],apl,T)
        # identify transition state
        G_TS = np.max(CNT_area(r_range[r_range>r_init[j]],p,coatings[j],apl,T))
        # calculate activation barrier to embolism in kT
        delta_G = G_TS - gibbs

        # prefactor = 0.5*kb*T/h

        # if there is no barrier to embolism, it occurs spontaneously
        if delta_G <= 0:
            emb[j] = 1
        # otherwise, assume a 50% chance of crossing the barrier once on top of it
        else:
            emb[j] = 0.5*np.exp(-1*delta_G)

    return np.sum(emb)/n_bubble


if __name__ == "__main__":
    ####---------------------------Physical Parameters---------------------------####

    # Boltzmann constant
    kb = 1.38e-23  # J/K
    # Planck constant
    h = 6.62607e-34 # J s

    T = 300.0 # K
    p = -2e6 # Pa

    ####---------------------------Lognormal Parameters---------------------------####

    # 'equilibrium' area per lipid
    apl = 0.6 # nm^-2

    # mean bubble radius
    mean_r = 190 # nm
    mu = np.log(mean_r)

    # standard deviation in natural log units
    sigma = 0.6

    # number of bubbles in sample
    n_bubble = 5000

    # lets not go crazy here
    assert mu - 3*sigma > 1

    # radii values to evaluate the Gibbs free energy at
    r_range = np.logspace(
        start=mu - 3*sigma,
        stop=mu + 6*sigma,
        base=np.e,
        num=500
        )

    # single calculation of embolism probability for a conduit
    emb = probability(p, T, mu, sigma, apl, r_range/1e9, n_bubble)

    # print(emb)

    # scan across range of P
    p = np.linspace(-5e5, -5e6, 450)

    # set up output matrix
    out = np.zeros(shape=(450, 2))
    out[:,0] = p

    prob = np.zeros(450)

    for i in tqdm(range(450)):
        out[i,1] = probability(p[i], T, mu, sigma, apl, r_range/1e9, n_bubble)

    np.savetxt('embolism.dat', out, fmt='%.5f')