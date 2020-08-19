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
	self.no_act = 0
	self.nv_act = 0

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
                e_scf_mo_2 += 2*self.twoelecint_mo[i][j][i][j] - self.twoelecint_mo[i][i][j][j]
     
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

    def get_denom_Sv(self):

	self.Dv = np.zeros((self.nocc,self.nv_act,self.nvirt,self.nvirt))
        for i in range(0,self.nocc):
            for c in range(0,self.nv_act):
                for a in range(0,self.nvirt):
                    for b in range(0,self.nvirt):
                        self.Dv[i,c,a,b] = self.mo_energy[i] - self.mo_energy[c+self.nocc] - self.mo_energy[a+self.nocc] - self.mo_energy[b+self.nocc]

    def get_denom_So(self):

	self.Do = np.zeros((self.nocc,self.nocc,self.nvirt,self.no_act))
        for i in range(0,self.nocc):
            for j in range(0,self.nocc):
                for a in range(0,self.nvirt):
                    for k in range(self.nocc-self.no_act,self.nocc):
                        self.Do[i,j,a,k-self.nocc+self.no_act] = self.mo_energy[i] + self.mo_energy[j] - self.mo_energy[a+self.nocc] + self.mo_energy[k]

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

    def get_tau(self, rank_t1):

	self.tau = cp.deepcopy(self.t2)
	if (rank_t1 > 1):
	    self.tau += np.einsum('ia,jb->ijab', self.t1, self.t1)

    def init_guess_Sv(self):
	self.get_denom_Sv()

	self.Sv = np.zeros((self.nocc,self.nv_act,self.nvirt,self.nvirt))
        for i in range(0,self.nocc):
            for c in range(0,self.nv_act):
                for a in range(0,self.nvirt):
                    for b in range(0,self.nvirt):
          		self.Sv[i,c,a,b] = self.twoelecint_mo[i,c+self.nocc,a+self.nocc,b+self.nocc]/self.Dv[i,c,a,b]

    def init_guess_So(self):
	self.get_denom_So()

	self.So = np.zeros((self.nocc,self.nocc,self.nvirt,self.no_act))
        for i in range(0,self.nocc):
            for j in range(0,self.nocc):
                for a in range(0,self.nvirt):
                    for k in range(self.nocc-self.no_act,self.nocc):
          		self.So[i,j,a,k-self.nocc+self.no_act] = self.twoelecint_mo[i,j,a+self.nocc,k]/self.Do[i,j,a,k-self.nocc+self.no_act]


    ### Setup DIIS
    def diis_ini(self, A):
        diis_vals_A = [A.copy()]
        diis_errors = []

        return diis_vals_A, diis_errors

    def init_diis_t1(self):
	self.diis_vals_t1, self.diis_errors_t1 = self.diis_ini(self.t1)

    def init_diis_t2(self):
	self.diis_vals_t2, self.diis_errors_t2 = self.diis_ini(self.t2)

    def init_diis_So(self):
	self.diis_vals_So, self.diis_errors_So = self.diis_ini(self.So)

    def init_diis_Sv(self):
	self.diis_vals_Sv, self.diis_errors_Sv = self.diis_ini(self.Sv)

    # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
    def diis_error_matrix(self, diis_size):

        B = np.ones((diis_size + 1, diis_size + 1)) * -1
        B[-1, -1] = 0
        for n1, e1 in enumerate(self.diis_errors):
            for n2, e2 in enumerate(self.diis_errors):
                # Vectordot the error vectors
                B[n1, n2] = np.dot(e1, e2)
        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
        # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
        resid = np.zeros(diis_size + 1)
        resid[-1] = -1
        print resid
        # Solve Pulay equations, [Pulay:1980:393], Eqn. 6
        self.ci = np.linalg.solve(B, resid)

    def new_amp(self, A, diis_size, ci, diis_vals_A): 
        A[:] = 0
        for num in range(diis_size):
          A += ci[num] * diis_vals_A[num + 1]
        return A

    def update_diis_t1(self, diis_size):
	self.t1 = self.new_amp(self.t1, diis_size, self.ci, self.diis_vals_t1)

    def update_diis_t2(self, diis_size):
	self.t2 = self.new_amp(self.t2, diis_size, self.ci, self.diis_vals_t2)

    def update_diis_So(self, diis_size):
	self.So = self.new_amp(self.So, diis_size, self.ci, self.diis_vals_So)

    def update_diis_Sv(self, diis_size):
	self.Sv = self.new_amp(self.Sv, diis_size, self.ci, self.diis_vals_Sv)

    def errors_diis_t1(self):
        self.diis_vals_t1.append(self.t1.copy())
        error_t1 = (self.t1 - self.old_t1).ravel()

	return error_t1

    def errors_diis_t2(self):
        self.diis_vals_t2.append(self.t2.copy())
        error_t2 = (self.t2 - self.old_t2).ravel()

	return error_t2

    def errors_diis_So(self):
        self.diis_vals_So.append(self.So.copy())
        error_So = (self.So - self.old_So).ravel()

	return error_So

    def errors_diis_Sv(self):
        self.diis_vals_Sv.append(self.Sv.copy())
        error_Sv = (self.Sv - self.old_Sv).ravel()

	return error_Sv

    ### All DIIS related functions are done

    def symmetrize(self,R_ijab):
        R_ijab_new = np.zeros((self.nocc,self.nocc,self.nvirt,self.nvirt))
        for i in range(0,self.nocc):
            for j in range(0,self.nocc):
                for a in range(0,self.nvirt):
                    for b in range(0,self.nvirt):
                        R_ijab_new[i,j,a,b] = R_ijab[i,j,a,b] + R_ijab[j,i,b,a]
    
        R_ijab = cp.deepcopy(R_ijab_new)
 
        return R_ijab
        R_ijab = None
        R_ijab_new = None

    def update_t1_t2(self, R_ia, R_ijab):
        ntmax = 0
        eps = 100

	self.old_t2 = self.t2.copy()
	self.old_t1 = self.t1.copy()

        delt2 = np.divide(R_ijab,self.D2)
        delt1 = np.divide(R_ia,self.D1)
        self.t1 = self.t1 + delt1
        self.t2 = self.t2 + delt2
        ntmax = np.size(self.t1)+np.size(self.t2)
        eps = float(np.sum(abs(R_ia)+np.sum(abs(R_ijab)))/ntmax)
	delt1 = None
	delt2 = None
        return eps

    def update_t2(self, R_ijab):
        ntmax = 0
        eps = 100

	self.old_t2 = self.t2.copy()

        delt2 = np.divide(R_ijab,self.D2)
        self.t2 = self.t2 + delt2
        ntmax = np.size(self.t2)
        eps = float(np.sum(abs(R_ijab))/ntmax)
	delt2 = None
	#print R_ijab
        return eps

    def update_So(self, R_ijav):
        ntmax = 0
        eps = 100

	self.old_So = self.So.copy()

        delSo = np.divide(R_ijav,self.Do)
        self.So = self.So + delSo
        ntmax = np.size(self.So)
        eps = float(np.sum(abs(R_ijav))/ntmax)
	delSo = None
        return eps

    def update_Sv(self, R_iuab):
        ntmax = 0
        eps = 100

	self.old_Sv = self.Sv.copy()

        delSv = np.divide(R_iuab,self.Dv)
        self.Sv = self.Sv + delSv
        ntmax = np.size(self.Sv)
        eps = float(np.sum(abs(R_iuab))/ntmax)
	delSv = None
        return eps

