import numpy as np

from pyscf.lib import logger

from PostHF import GetIntNData

import utils


class state(object):
    def __init__(self, mf, variant = 'CCSD', nfo=0, nfv=0):

        self.mol  = mf.mol
        self.mf = mf

        self.nfo = nfo
        self.nfv = nfv

	self.nel = self.mol.nelectron

	self.e_hf = mf.e_tot

	self.variant = variant
	self.tInitParams = True
	self.tdiis = True
	
	self.no_act = 0
	self.nv_act = 0
	
	self.maxiter = 50
	self.max_diis = 20
	self.conv = 1e-7

    def init_parameters(self):
	
	self.rank_So = 0
	self.rank_Sv = 0
	if (self.variant == 'LCCD'):
	    self.rank_t1 = 0
	    self.rank_t2 = 1
	    print '**** RUNNING LCCD ****'
	elif (self.variant == 'CCD'):
	    self.rank_t1 = 0
	    self.rank_t2 = 2
	    print '**** RUNNING CCD ****'
	elif (self.variant == 'CCSD'):
	    self.rank_t1 = 4
	    self.rank_t2 = 2
	    print '**** RUNNING CCSD ****'
	elif (self.variant == 'ICCSD'):
	    self.rank_t1 = 4
	    self.rank_t2 = 2
	    self.rank_So = 1
	    self.rank_Sv = 1
	    print '**** RUNNING ICCSD ****'

	self.tInitParams = False
	self.e_old = self.e_hf

    def init_amplitudes(self, data):

	data.init_guess_t2()

	if (self.rank_t1 > 0):
	    data.init_guess_t1()

	if (self.rank_So > 0):
	    data.init_guess_So()

	if (self.rank_Sv > 0):
	    data.init_guess_Sv()

	data.get_tau(self.rank_t1)


    def init_diis(self, data):

	if (self.rank_t1 > 0):
	    data.init_diis_t1()

	if (self.rank_t2 > 0):
	    data.init_diis_t2()

	if (self.rank_So > 0):
	    data.init_diis_So()

	if (self.rank_Sv > 0):
	    data.init_diis_Sv()

	data.diis_errors = []

    def update_diis(self, data, x):

        # Limit size of DIIS vector
        if (len(data.diis_vals_t2) > self.max_diis):
	    if (self.rank_t1 > 0):
	        del data.diis_vals_t1[0]
	    if (self.rank_t2 > 0):
	        del data.diis_vals_t2[0]
	    if (self.rank_So > 0):
	        del data.diis_vals_So[0]
	    if (self.rank_Sv > 0):
	        del data.diis_vals_Sv[0]
	    del data.diis_errors[0]
	self.diis_size = len(data.diis_vals_t2) - 1

        # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
        ci = data.diis_error_matrix(self.diis_size)
        # Calculate new amplitudes
        if (x+1) % self.max_diis == 0:
	    if (self.rank_t1 > 0):
	        data.update_diis_t1(self.diis_size)
	    if (self.rank_t2 > 0):
	        data.update_diis_t2(self.diis_size)
	    if (self.rank_So > 0):
	        data.update_diis_So(self.diis_size)
	    if (self.rank_Sv > 0):
	        data.update_diis_Sv(self.diis_size)

        # End DIIS amplitude update
    

    def energy_cc(self, data):
	occ = data.nocc
	nao = data.nao
        e_cc  = 2*np.einsum('ijab,ijab',data.t2,data.twoelecint_mo[:occ,:occ,occ:nao,occ:nao]) 
        e_cc += -np.einsum('ijab,ijba',data.t2,data.twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
	if (self.rank_t1 > 0):
            e_cc += 2*np.einsum('ijab,ia,jb',data.twoelecint_mo[:occ,:occ,occ:nao,occ:nao],data.t1,data.t1) 
            e_cc += - np.einsum('ijab,ib,ja',data.twoelecint_mo[:occ,:occ,occ:nao,occ:nao],data.t1,data.t1)
        return e_cc

    def convergence(self, e_cc, e_old, eps, x):
        del_e = e_cc - e_old
        if abs(eps) <= self.conv and abs(del_e) <= self.conv:
            print "ccsd converged!!!"
            print "Total energy is : "+str(self.e_hf + e_cc)
            return True
        else:
            print "cycle number : "+str(x+1)
            print "change in t1 and t2 : "+str(eps)
            print "energy difference : "+str(del_e)
            print "energy : "+str(self.e_hf + e_cc)
            return False

    def convergence_ext(self, e_cc, e_old, eps, eps_So, eps_Sv, x):
        del_e = e_cc - e_old
        if abs(eps) <= self.conv and abs(eps_So) <= self.conv and abs(eps_Sv) <= self.conv and abs(del_e) <= self.conv:
            print "change in t1+t2 , So, Sv : "+str(eps)+" "+str(eps_So)+" "+str(eps_Sv)
            print "energy difference : "+str(del_e)
            print "ccsd converged!!!"
            print "Total energy is : "+str(self.e_hf + e_cc)
            return True
        else:
    	    print "cycle number : "+str(x+1)
            print "change in t1+t2 , So, Sv : "+str(eps)+" "+str(eps_So)+" "+str(eps_Sv)
            print "energy difference : "+str(del_e)
            print "energy : "+str(self.e_hf + e_cc)
            return False

    def calc_residue(self, data):

	intermediates = utils.intermediates(data)
	amplitude = utils.amplitude(data)

	# First get all the intermediates
        I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2 = intermediates.initialize()
	if (self.rank_t2 > 1):
            I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov = intermediates.update_int(I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov)

	if (self.rank_t1 > 0):
	    I1, I2 = intermediates.R_ia_intermediates()

	if (self.rank_So > 0):
	    II_oo = intermediates.W1_int_So()
	    II_vv = intermediates.W1_int_Sv()
	    II_ov = intermediates.coupling_terms_So()
	    II_vo = intermediates.coupling_terms_Sv()

	    II_ovoo,II_ovoo3,II_vvvo3 = intermediates.W2_int_So()
            II_vvvo,II_vvvo2,II_ovoo2 = intermediates.W2_int_Sv()

	if (self.rank_t1 > 0):
	    self.R_ia = amplitude.singles(I1,I2,I_oo,I_vv)
            I_oo,I_vv,I_oovo,I_vovv,Ioooo_2,I_voov,Iovov_3,Iovvo_3,Iooov,I3=intermediates.singles_intermediates(I_oo,I_vv,I2, self.rank_t1)

        self.R_ijab = amplitude.doubles(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2)

        if (self.rank_t1 > 0):
            self.R_ijab += amplitude.singles_n_doubles(I_oovo,I_vovv, self.rank_t1)
            self.R_ijab += amplitude.higher_order(Iovov_3, Iovvo_3, Iooov, I3, Ioooo_2, I_voov, self.rank_t1)

	if (self.rank_So > 0):
            self.R_ijab += amplitude.inserted_diag_So(II_oo) 

            self.R_ijav = amplitude.So_diagram_vs_contraction()
            self.R_ijav += amplitude.So_diagram_vt_contraction()
            self.R_ijav += amplitude.v_sv_t_contraction_diag(II_vo)
            self.R_ijav += amplitude.w2_diag_So(II_ovoo,II_vvvo2,II_ovoo2)
            if (self.rank_t1 > 0):
                self.R_ijav += amplitude.T1_contribution_So()
	        self.R_ia += amplitude.inserted_diag_So_t1(II_oo)

	if (self.rank_Sv > 0):
            self.R_ijab += amplitude.inserted_diag_Sv(II_vv) 

            self.R_iuab = amplitude.Sv_diagram_vs_contraction()
            self.R_iuab += amplitude.Sv_diagram_vt_contraction()
            self.R_iuab += amplitude.v_so_t_contraction_diag(II_ov)
            self.R_iuab += amplitude.w2_diag_Sv(II_vvvo,II_ovoo3,II_vvvo3)
            if (self.rank_t1 > 0):
                self.R_iuab += amplitude.T1_contribution_Sv()
	        self.R_ia += amplitude.inserted_diag_Sv_t1(II_vv)
    

        self.R_ijab = data.symmetrize(self.R_ijab)

	    
    def update_amplitudes(self, data):

	if (self.rank_t2 > 0 and self.rank_t1 > 0):
	    self.eps = data.update_t1_t2(self.R_ia, self.R_ijab)
	else:
	    self.eps = data.update_t2(self.R_ijab)

	if (self.rank_So > 0):
	    self.eps_So = data.update_So(self.R_ijav)

	if (self.rank_Sv > 0):
	    self.eps_Sv = data.update_Sv(self.R_iuab)

	
    def converge_cc_eqn(self, data):

	for x in range(0, self.maxiter):

	    data.get_tau(self.rank_t1)
	    self.calc_residue(data)

	    self.update_amplitudes(data)
	    if ((x+1) > self.max_diis) and self.tdiis:
		self.update_diis(data, x)

	    e_cc = self.energy_cc(data)

            if (self.rank_So > 0):
                val = self.convergence_ext(e_cc,self.e_old,self.eps, self.eps_So, self.eps_Sv, x)
	    else: 
                val = self.convergence(e_cc,self.e_old,self.eps, x)

            if val == True :
                break
            else:  
                self.e_old = e_cc
	
	    if self.tdiis:
	        if (self.rank_t1 > 0):
	            errors_t1 = data.errors_diis_t1()
	        if (self.rank_t2 > 0):
	            errors_t2 = data.errors_diis_t2()
	        if (self.rank_So > 0):
	            errors_So = data.errors_diis_So()
	        if (self.rank_Sv > 0):
	            errors_Sv = data.errors_diis_Sv()
		
	    if (self.rank_t2 > 0 and self.rank_t1 > 0):
	        data.diis_errors.append(np.concatenate((errors_t1,errors_t2)))
	    else:
	        data.diis_errors.append((errors_t2))

    def run(self):


	AllData = GetIntNData(self.mf, self.nfo, self.nfv)

	AllData.transform_all_ints()

	self.twoelecint_mo = AllData.twoelecint_mo

	self.nao = AllData.nao
	self.nocc = AllData.nocc
	self.nvirt = AllData.nvirt

	AllData.no_act = self.no_act
	AllData.nv_act = self.nv_act

	if self.tInitParams:
	    self.init_parameters()

	self.init_amplitudes(AllData)

	if self.tdiis:
	    self.init_diis(AllData)

	self.converge_cc_eqn(AllData)


        print '**** CCSD is done ****'
