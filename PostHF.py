import numpy as np

from pyscf import scf
from pyscf import symm
from pyscf.lib import logger
import copy as cp


class GetIntNData(object):
    def __init__(self, mf, nfo=0, nfv=0, mo_coeff=None, mo_occ=None):

	if mo_coeff is None: mo_coeff = mf.mo_coeff
	if mo_occ   is None: mo_occ   = mf.mo_occ

        self.mol  = mf.mol
	self._scf = mf
	self.verbose = self.mol.verbose
	self.stdout = self.mol.stdout

        self.nfo = nfo
        self.nfv = nfv

        self.mo_energy = mf.mo_energy
	self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.e_corr = None
        self.t2 = None
        self._keys = set(self.__dict__.keys())
        
        self.oneelecint_mo = None
        self.twoelecint_mo = None
        self.fock_mo = None

	self.e_hf = mf.e_tot

	self.orb_sym = []

        # Obtain the number of atomic orbitals in the basis set
        self.nao = self.mol.nao_nr()
        # Obtain the number of electrons
        self.nel = self.mol.nelectron
        # Compute nuclear repulsion energy
        self.enuc = self.mol.energy_nuc()

	if (self.nel%2 == 0):
	    self.nocc = self.nel/2
	else:
	    print ('can not handle open shell cases: Quitting...')
	    quit()
	
	self.nvirt = self.nao - self.nocc

    def init_integrals(self):
        # Compute one-electron kinetic integrals
        self.T = self.mol.intor('cint1e_kin_sph')
        # Compute one-electron potential integrals
        self.V = self.mol.intor('cint1e_nuc_sph')
        # Compute one-electron total integrals
	self.F = self.T + self.V
        # Compute two-electron repulsion integrals (Chemists' notation)
        self.v2e = self.mol.intor('cint2e_sph').reshape((self.nao,)*4)

    def transform_1e_ints(self):
    
        ##--------------------------------------------------------------##
              #Transform the 1 electron integral to MO basis#
        ##--------------------------------------------------------------##
        
        self.oneelecint_mo = np.einsum('ab,ac,cd->bd', self.mo_coeff, self.F, self.mo_coeff)
    
    def transform_2e_ints(self):
    
        twoelecint_1 = np.einsum('zs,wxyz->wxys', self.mo_coeff, self.v2e)
        twoelecint_2 = np.einsum('yr,wxys->wxrs', self.mo_coeff, twoelecint_1)
        twoelecint_1 = None
        twoelecint_3 = np.einsum('xq,wxrs->wqrs', self.mo_coeff, twoelecint_2)
        twoelecint_2 = None
        twoelecint_4 = np.einsum('wp,wqrs->pqrs', self.mo_coeff, twoelecint_3)
        twoelecint_3 = None
        self.twoelecint_mo = np.swapaxes(twoelecint_4,1,2)
        twoelecint_4 = None
    
        if self.nfo > 0:
            self.twoelecint_mo = cp.deepcopy(self.twoelecint_mo[self.nfo:,self.nfo:,self.nfo:,self.nfo:])

        if self.nfv > 0:
            self.twoelecint_mo = cp.deepcopy(self.twoelecint_mo[:-self.nfv,:-self.nfv,:-self.nfv,:-self.nfv])

    def calc_energy_mo(self):
    
        e_scf_mo_1 = 0.0
        e_scf_mo_2 = 0.0
    
        for i in range(0,self.nocc):
            e_scf_mo_1 += self.oneelecint_mo[i][i]
    
        for i in range(0,self.nocc):
            for j in range(0,self.nocc):
                e_scf_mo_2 += 2*self.twoelecint_mo[i][i][j][j] - self.twoelecint_mo[i][j][i][j]
     
        e_scf_mo = 2*e_scf_mo_1 + e_scf_mo_2 + self.enuc

	return e_scf_mo
    
    def GetFock(self):

        self.fock_mo = np.zeros((self.nao,self.nao))
	for i in range(0,self.nao):
	    self.fock_mo[i,i] = self.mo_energy[i]

        if self.nfo > 0:
            self.fock_mo = cp.deepcopy(self.fock_mo[self.nfo:,self.nfo:])

        if self.nfv > 0:
            self.fock_mo = cp.deepcopy(self.fock_mo[:-self.nfv,:-self.nfv])

    def apply_frozen_orb(self):

        if self.nfo > 0:
            self.mo_energy = cp.deepcopy(self.mo_energy[self.nfo:])

            self.nocc = self.nocc - self.nfo
            self.nao = self.nao - self.nfo

        if self.nfv > 0:
            self.mo_energy = self.mo_energy[:-self.nfv]

            self.nao = self.nao - self.nfv
            self.nvirt = self.nvirt - self.nfv  

    def transform_all_ints(self):
    
	self.init_integrals()

        self.transform_1e_ints()
        self.transform_2e_ints()
    
        e_scf_mo = self.calc_energy_mo()
    
        if abs(e_scf_mo - self.e_hf)<= 1E-6 :
            print "MO conversion successful"
    	else: 
    	    print "MO conversion not successful"
    
	self.GetFock()

	self.apply_frozen_orb()

    def get_orb_sym(self):

	mo = symm.symmetrize_orb(self.mol, self.mo_coeff)
        self.orb_sym = symm.label_orb_symm(self.mol, self.mol.irrep_id, self.mol.symm_orb, mo)

        if self.nfo > 0:
            self.orb_sym = self.orb_sym[self.nfo:]

    def get_denom_t1(self):

        self.D1 = np.zeros((self.nocc,self.nvirt))
        for i in range(0,self.nocc):
            for a in range(self.nocc,self.nao):
                self.D1[i,a-self.nocc] = self.mo_energy[i] - self.mo_energy[a]

    def get_denom_t2(self):

	self.D2 = np.zeros((self.nocc,self.nocc,self.nvirt,self.nvirt))
        for i in range(0,self.nocc):
            for j in range(0,self.nocc):
                for a in range(self.nocc,self.nao):
                    for b in range(self.nocc,self.nao):
                        self.D2[i,j,a-self.nocc,b-self.nocc] = self.mo_energy[i] + self.mo_energy[j] - self.mo_energy[a] - self.mo_energy[b]

    def init_guess_t1(self):
	self.get_denom_t1()

        self.t1 = np.zeros((self.nocc,self.nvirt))
        for i in range(0,self.nocc):
            for a in range(self.nocc,self.nao):
                self.t1[i,a-self.nocc] = self.fock_mo[i,a]/self.D1[i,a-self.nocc]

    def init_guess_t2(self):
	self.get_denom_t2()
	self.t2 = np.zeros((self.nocc,self.nocc,self.nvirt,self.nvirt))
        for i in range(0,self.nocc):
            for j in range(0,self.nocc):
                for a in range(self.nocc,self.nao):
                    for b in range(self.nocc,self.nao):
                        self.t2[i,j,a-self.nocc,b-self.nocc] = self.twoelecint_mo[i,j,a,b]/self.D2[i,j,a-self.nocc,b-self.nocc]


    def run(self):

	log = logger.Logger(self.stdout, self.verbose)

	#log.warn('**** RUNING MP2 ****')

