import os,sys
import torch
import numpy as np
from scipy.stats import norm
from botorch.acquisition.analytic import UpperConfidenceBound, LogExpectedImprovement

class optim(object):
    def __init__(self,surrogate,input_space,method,preference = None):
        self.model = surrogate
        self.x     = input_space
        self.method= method
        self.prefm = preference

def acquire(opt, args = None):
    
    methods = {
        'random' : random,
        'ucb'    : ucb,
        'ei'     : ei,
        'pibo'   : pibo,
        'coexbo' : coexbo
    }
    
    idx = methods[opt.method](opt,args)

    return idx

def random(opt,args):

    all_idxs = np.arange(opt.x.shape[0])
    idx = np.random.choice(all_idxs)

    return idx

def ucb(opt,args):

    beta = float(args.beta)
    mu    = opt.model.predict(opt.x).flatten()
    stdev = np.sqrt(opt.model.variance(opt.x)).flatten()
    ucb_vals = mu + beta*stdev
    idx = np.argmax(ucb_vals)
    
    return idx

def ei(opt,args, pibo = False, d = 0.01):

    fmax = float(args.fmax)
    i = opt.model.predict(opt.x).flatten() - fmax - d
    s = np.sqrt(opt.model.variance(opt.x).flatten())
    z = i / s
    pdf_values = norm.pdf(z).flatten()
    cdf_values = norm.cdf(z).flatten()
    ei_vals = i*cdf_values + s*pdf_values
    ei_vals[np.where(s <= d)] = 0

    if pibo:
        return ei_vals

    idx = np.argmax(ei_vals)

    return idx

def pibo(opt,args):

    #compute the expected improvement part of the AQF
    ei_vals = ei(opt,args, pibo = True, d=0.0)

    #compute the user preference portion of the acquisition function
    beta = float(args.beta)
    n    = float(args.iter)
    exp  = beta / n
    pi = (opt.prefm**exp).flatten()

    #select the candidate
    acf = ei_vals*pi
    idx = np.argmax(acf)

    return idx

def coexbo(opt,args):
    
    #constants
    t     = float(args.iter)
    gamma = 0.01
    beta  = 0.01
    #intermediate values
    mu_f  = opt.model.predict(opt.x).flatten()
    var_f = opt.model.variance(opt.x).flatten()
    mu_pi = opt.prefm
    var_pi= opt.prefv + gamma*(t**2)*var_f
    comb_var = var_pi*var_f / (var_pi + var_f)
    comb_mu  = comb_var / var_pi * mu_pi + comb_var / var_f * mu_f
    #aqf
    comb_ucb = comb_mu + beta*comb_var
    idx = np.argmax(comb_ucb)

    return idx
