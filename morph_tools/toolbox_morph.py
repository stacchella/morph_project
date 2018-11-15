# coding: utf-8

#!/usr/bin/env python2

"""
Created on November 14, 2018

@author: sandro.tacchella@cfa.harvard.edu

"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo


def make_SFR_profile(radius, SFR_total, Rs):
    '''
    This function returns the SFR surface density profile,
    which we assume to be an exponential disk.
    Input:
        radius    : in kpc
        SFR_total : total SFR (in Msun/yr)
        Rs        : scale radius (in kpc)
    Output:
        SFRD      : SFR surface density (in Msun/yr/kpc^2)
    '''
    normalization = SFR_total/(2.0*np.pi*Rs**2)
    return(normalization*np.exp(-1.0*radius/Rs))


def get_size_from_profile(radius, profile):
    '''
    This function returns the half-mass size of a
    2d surface density profile.
    Input:
        radius     : radial coordinates of profile
        profile    : surface density profile
    Output:
        size       : half-mass radius
    '''
    # calculate cumulative
    cumulative_from_profile = []
    for rii in radius:
        idx = (radius <= rii)
        cumulative_from_profile = np.append(cumulative_from_profile, np.trapz(2.0*np.pi*radius[idx]*profile[idx], radius[idx]))
    # get half mass value
    half_mass = 0.5*cumulative_from_profile[-1]
    return(np.interp(half_mass, cumulative_from_profile, radius))



class galaxy(object):
    
    def __init__(self, radius, list_scale_factor, list_SFRtot, Rs_params):
        self.radius = np.array(radius)
        self.scale_factor = np.array(list_scale_factor)
        self.redshift = 1.0/np.array(list_scale_factor)-1
        self.time = cosmo.age(self.redshift).value
        self.scale_factor_boundary = np.append(0.0, self.scale_factor+0.5*np.append(np.diff(self.scale_factor), np.diff(self.scale_factor)[-1]))
        self.time_boundary = cosmo.age(1.0/self.scale_factor_boundary-1.0).value
        self.time_dt = np.diff(cosmo.age(1.0/self.scale_factor_boundary-1.0).value)
        self.SFR = np.array(list_SFRtot)
        self.mass = np.cumsum(self.time_dt*10**9*self.SFR)
        self.Rs_params = np.array(Rs_params)
        self.Rs = self.Rs_params[0]*(self.mass/10**10)**self.Rs_params[1]*((self.SFR/self.mass)/10**-10)**self.Rs_params[2]*(1+self.redshift)**self.Rs_params[3]
        self.age = 0.5*self.time_dt[::-1] + np.append(0.0, np.cumsum(self.time_dt[::-1])[:-1])
    
    def build_mass_profile(self):
        profile_collection = np.zeros((len(self.time_dt), len(self.radius)))
        for ii in np.arange(len(self.time_dt)):
            profile_collection[ii] = self.time_dt[ii]*make_SFR_profile(self.radius, self.SFR[ii], self.Rs[ii])
        return(profile_collection)
    
    def get_age_profile(self):
        profile_collection = self.build_mass_profile()
        profile_age = np.sum((profile_collection.T*self.age[::-1]).T, axis=0)/np.sum(profile_collection, axis=0)
        return(profile_age)
        
    def get_mass_profile(self):
        profile_collection = self.build_mass_profile()
        return(np.sum(profile_collection, axis=0))
    
    def get_size(self):
        profile_collection = self.build_mass_profile()
        profile_collection_cumsum = np.cumsum(profile_collection, axis=0)
        RM = []
        for ii in range(len(self.time_dt)):
            RM.append(get_size_from_profile(self.radius, profile_collection_cumsum[ii]))
        return(np.array(RM))
    
    
    
