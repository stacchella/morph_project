# coding: utf-8

#!/usr/bin/env python2

"""
Created on November 14, 2018

@author: sandro.tacchella@cfa.harvard.edu

"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
import os


path_main = os.environ['SRMP_PATH']

t_mass = Table.read(path_main + 'data/mass_loss_table.dat', format='ascii')


def get_mass_fraction(time_list):
    mass_fraction = []
    for ii_t in time_list:
        mass_fraction = np.append(mass_fraction, np.interp(ii_t, t_mass['time'], t_mass['mass_frac']))
    return(np.array(mass_fraction))


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
    if ((SFR_total == 0.0) | (Rs == 0.0)):
        return(np.zeros(len(radius)))
    else:
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


def get_mass_within_radius(radius, profile, Rmax):
    '''
    This function returns the half-mass size of a
    2d surface density profile.
    Input:
        radius     : radial coordinates of profile
        profile    : surface density profile
        Rmax       : maximal radius
    Output:
        mass       : mass within Rmax
    '''
    # calculate cumulative
    cumulative_from_profile = []
    for rii in radius:
        idx = (radius <= rii)
        cumulative_from_profile = np.append(cumulative_from_profile, np.trapz(2.0*np.pi*radius[idx]*profile[idx], radius[idx]))
    # get half mass value
    mass_inR = np.interp(Rmax, radius, cumulative_from_profile)
    return(mass_inR)



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
        self.Rs = self.Rs_params[0]*(self.mass/10**10)**self.Rs_params[1]*(1+self.redshift)**self.Rs_params[2]  # ((self.SFR/self.mass)/10**-10)**self.Rs_params[2]
        self.Rs[np.isnan(self.Rs)] = 0.0
        self.age = 0.5*self.time_dt[::-1] + np.append(0.0, np.cumsum(self.time_dt[::-1])[:-1])
    
    def build_mass_profile(self):
        profile_collection = np.zeros((len(self.time_dt), len(self.radius)))
        for ii in np.arange(len(self.time_dt)):
            profile_collection[ii] = 10**9*self.time_dt[ii]*make_SFR_profile(self.radius, self.SFR[ii], self.Rs[ii])
        return(profile_collection)
    
    def get_age_profile(self):
        profile_collection = self.build_mass_profile()
        profile_age = np.sum((profile_collection.T*self.age[::-1]).T, axis=0)/np.sum(profile_collection, axis=0)
        return(profile_age)
        
    def get_mass_profile(self):
        profile_collection = self.build_mass_profile()
        return(np.sum(profile_collection, axis=0))
    
    def get_size(self, redshift_in=np.nan):
        profile_collection = self.build_mass_profile()
        profile_collection_cumsum = np.cumsum(profile_collection, axis=0)
        if np.isfinite(redshift_in):
            idx = (np.abs(self.redshift - redshift_in)).argmin()
            return(get_size_from_profile(self.radius, profile_collection_cumsum[idx]))
        else:
            RM = []
            for ii in range(len(self.time_dt)):
                RM.append(get_size_from_profile(self.radius, profile_collection_cumsum[ii]))
            return(np.array(RM))
    
    def get_mass_within_R(self, Rmax, redshift_in=np.nan):
        profile_collection = self.build_mass_profile()
        profile_collection_cumsum = np.cumsum(profile_collection, axis=0)
        if np.isfinite(redshift_in):
            idx = (np.abs(self.redshift - redshift_in)).argmin()
            return(get_mass_within_radius(self.radius, profile_collection_cumsum[idx], Rmax))
        else:
            M = []
            for ii in range(len(self.time_dt)):
                M.append(get_mass_within_radius(self.radius, profile_collection_cumsum[ii], Rmax))
            return(np.array(M))
    
    def get_mass_after_mass_loss(self, redshift_in):
        idx = (np.abs(self.redshift - redshift_in)).argmin()
        weights = get_mass_fraction(self.time[-1]-self.time)
        return(np.cumsum(weights[:idx]*self.time_dt[:idx]*10**9*self.SFR[:idx])[-1])

    
    
