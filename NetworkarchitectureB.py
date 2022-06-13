#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:22:02 2018
@author: ksxu
"""
import numpy as np

class Networkmodel(object):
    def __init__(self, Ne1, Ni1, cp, W_gap, W_Chem1, W_ei1, W_ii1, W_ie1):
        """# input the parameters
        Ne1, Ni1 : The numbers of excitatroy and inbihibtory ensemble,respectively.
        cp: the probabilty to add links
        W_gap: the weight of gap junction for excitatoty population.
        W_Chem1: the weight of chemical synapse for excitatoty population.
        W_ei1: the weight of inhibitory synapse currents from inhibitory to excitatory population.
        W_ii1: the weight of inhibitory currents in inhibitory population from itself.
        W_ie1: the weight of excitatory currents in inhibitory population from excitatory population
         """
        self.Ne1 = Ne1
        self.Ni1 = Ni1
        self.cp = cp
        self.W_gap = W_gap
        self.W_Chem1 = W_Chem1
        self.W_ei1 = W_ei1
        self.W_ii1 = W_ii1
        self.W_ie1 = W_ie1
        
    def alltoall(self):
        # sub-population E1
        self.wE1E1_gap = self.W_gap * np.random.binomial(1,self.cp,(self.Ne1, self.Ne1))
        self.wE1E1_Chem = self.W_Chem1 * np.ones((self.Ne1, self.Ne1)) 
        self.wE1I1 = - self.W_ei1*np.ones((self.Ne1, self.Ni1))
        self.wE1E1_gap[np.diag_indices(self.Ne1)] = 0
        self.wE1E1_Chem[np.diag_indices(self.Ne1)] = 0
         
        
        #sub-population I1
        self.wI1I1_gap = self.W_gap * np.random.binomial(1,self.cp, (self.Ni1, self.Ni1))
        self.wI1I1_chem = - self.W_ii1*np.ones((self.Ni1, self.Ni1))
        self.wI1E1 = self.W_ie1*np.ones((self.Ni1, self.Ne1))
        self.wI1I1_gap[np.diag_indices(self.Ni1)] = 0
        self.wI1I1_chem[np.diag_indices(self.Ni1)] = 0
        return self.wE1E1_gap, self.wE1E1_Chem, self.wE1I1,self.wI1I1_chem,self.wI1E1
    
    def SmallWorld_test(self,deg=3):
        """smal-world topology
        Parameters:
        deg # the degree of the nodes for neural networks,default is 3
        
        Return
        """
        # the small work networks for excitatory population
        self.deg = deg
        CM = np.zeros((self.Ne1,self.Ne1))
        CM_chem = np.zeros((self.Ne1, self.Ne1))
        CM0 = np.zeros((self.Ne1, self.Ne1))
        CM[0,-1]=1
        for i in range(self.Ne1):
            for d in range(1,deg+1):
                CM[i,i-d]=1
                CM_chem[i,i-d]=1
            if np.random.uniform()<self.cp:
                r1=np.random.randint(i-self.Ne1+self.deg,i-self.deg)
                r2=np.random.randint(1,self.deg+1)
                CM[i,r1]=1
                if deg >= 2:
                    CM[i,i-r2]=0
        
        CM = CM+CM.T
        CM_chem = CM_chem+CM_chem.T
        
        DiffCM = CM-CM_chem
        DiffCM[DiffCM == -1] = 0
        CM0 = CM-DiffCM
        
        
        
        CM = np.minimum(CM,np.ones_like(CM))
        CM_chem = np.minimum(CM_chem,np.ones_like(CM_chem))
        DiffCM = np.minimum(DiffCM,np.ones_like(DiffCM))
        CM0 = np.minimum(CM0,np.ones_like(CM0))
        
        
        CMl = np.tril(CM)
        CMl_chem = np.tril(CM_chem)
        DiffCM = np.tril(DiffCM)
        CMl0 = np.tril(CM0)
        
        cc0=np.array(np.where(CMl0==1)).T
        cc=np.array(np.where(CMl==1)).T
        cc_chem=np.array(np.where(CMl_chem==1)).T
        cc_DiffCM=np.array(np.where(DiffCM==1)).T
        return cc0, cc, cc_chem, cc_DiffCM
    
    def SmallWorld(self,deg=3, opt = 0):
        """smal-world topology
        Parameters:
        deg # the degree of the nodes for neural networks,default is 3
        
        Return
        """
        self.deg = deg
        CM = np.zeros((self.Ne1,self.Ne1))
        CM_chem = np.zeros((self.Ne1, self.Ne1))
        CM_gap = np.zeros((self.Ne1, self.Ne1))
        CM0 = np.zeros((self.Ne1, self.Ne1))
        CM[0,-1]=1
                # the small work networks for excitatory population
        for s in range(1):
            np.random.seed(s+8)
            for i in range(self.Ne1):
                for d in range(1,self.deg+1):
                    CM_chem[i,i-d]=1 
                    CM_gap[i,i-d]=1
                    if np.random.uniform()<self.cp:
                        r1=np.random.randint(i-self.Ne1+self.deg,i)
                        CM[i,r1]=1
                    else:
                        CM[i,i-d]=1

        
        CM = CM+CM.T  # CM = CM+CM.T  #uncomment this to have a symmetrical network
        CM_chem = CM_chem+CM_chem.T
        CM_gap = CM_gap+CM_gap.T
        
        DiffCM = CM-CM_chem # the gap junction matrix in the small world networks
        DiffCM[DiffCM == -1] = 0
        CM0 = CM-DiffCM  #the chemical connection matrix in  the small world networks
        
        CM = np.minimum(CM,np.ones_like(CM))
        CM_chem = np.minimum(CM_chem,np.ones_like(CM_chem))
        DiffCM = np.minimum(DiffCM,np.ones_like(DiffCM))
        CM0 = np.minimum(CM0,np.ones_like(CM0))
        CM_gap = np.minimum(CM_gap,np.ones_like(CM_gap))
        
        if opt ==0 :
            # connection in the excitatory population
            self.wE1E1_gap = self.W_gap * DiffCM # the gap junction part of small world network
            self.wE1E1_Chem = self.W_Chem1 * CM0 # the chemical connection part of the small world network model 
            self.wE1I1 = self.W_ei1*np.ones((self.Ne1, self.Ni1))
            
            #the inhibitory population
            self.wI1I1_gap = self.W_gap * np.random.binomial(1,self.cp, (self.Ni1, self.Ni1))
            self.wI1I1_chem = self.W_ii1*np.ones((self.Ni1, self.Ni1))
            self.wI1E1 = self.W_ie1*np.ones((self.Ni1, self.Ne1))
            self.wI1I1_gap[np.diag_indices(self.Ni1)] = 0
            self.wI1I1_chem[np.diag_indices(self.Ni1)] = 0
        elif opt == 1:
            # connection in the excitatory population
            self.wE1E1_gap = self.W_gap *CM # the gap junction of small world network
            self.wE1E1_Chem = self.W_Chem1 * CM0 # the chemical connection part of the small world network model 
            self.wE1I1 = self.W_ei1*np.ones((self.Ne1, self.Ni1))
            
            #the inhibitory population
            self.wI1I1_gap = self.W_gap * np.random.binomial(1,self.cp, (self.Ni1, self.Ni1))
            self.wI1I1_chem = self.W_ii1*np.ones((self.Ni1, self.Ni1))
            self.wI1E1 = self.W_ie1*np.ones((self.Ni1, self.Ne1))
            self.wI1I1_gap[np.diag_indices(self.Ni1)] = 0
            self.wI1I1_chem[np.diag_indices(self.Ni1)] = 0
        else:
            #self.wE1E1_gap = self.W_gap *CM0  # the gap junction of small world network
            self.wE1E1_gap = self.W_gap * CM_gap # the gap junction of small world network
            self.wE1E1_Chem = self.W_Chem1 * CM # the chemical connection of the small world network model 
            self.wE1I1 = self.W_ei1*np.ones((self.Ne1, self.Ni1))
            
            #the inhibitory population
            self.wI1I1_gap = self.W_gap * np.random.binomial(1,self.cp, (self.Ni1, self.Ni1))
            self.wI1I1_chem = self.W_ii1*np.ones((self.Ni1, self.Ni1))
            self.wI1E1 = self.W_ie1*np.ones((self.Ni1, self.Ne1))
            self.wI1I1_gap[np.diag_indices(self.Ni1)] = 0
            self.wI1I1_chem[np.diag_indices(self.Ni1)] = 0

        return self.wE1E1_gap, self.wE1E1_Chem, self.wE1I1,self.wI1I1_chem,self.wI1E1
        
