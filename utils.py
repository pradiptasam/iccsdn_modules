import copy as cp
import numpy as np

class intermediates(object):
    def __init__(self, data):

	self.nao = data.nao
	self.nocc = data.nocc
	self.nvirt = data.nvirt
	self.no_act = data.no_act

	self.data = data
	self.nv_act = data.nv_act
	self.n_act = self.no_act + self.nv_act

	self.data = data


    def initialize(self):
	occ = self.nocc
	nao = self.nao
        I_vv = cp.deepcopy(self.data.fock_mo[occ:nao,occ:nao])
        I_oo = cp.deepcopy(self.data.fock_mo[:occ,:occ])
        Ivvvv = cp.deepcopy(self.data.twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao])
        Ioooo = cp.deepcopy(self.data.twoelecint_mo[:occ,:occ,:occ,:occ])
        Iovvo = cp.deepcopy(self.data.twoelecint_mo[:occ,occ:nao,occ:nao,:occ])
        Iovvo_2 = cp.deepcopy(self.data.twoelecint_mo[:occ,occ:nao,occ:nao,:occ])
        Iovov = cp.deepcopy(self.data.twoelecint_mo[:occ,occ:nao,:occ,occ:nao])
        Iovov_2 = cp.deepcopy(self.data.twoelecint_mo[:occ,occ:nao,:occ,occ:nao])
        return I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov, Iovov_2
     
        I_vv = None
        I_oo = None
        Ivvvv = None
        Ioooo = None
        Iovvo = None
        Iovvo_2 = None
        Iovov = None
        Iovov_2 = None
        I_oovo = None
        I_vovv = None
        gc.collect()

    def update_int(self,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov):

	occ = self.nocc
	nao = self.nao
        I_vv += -2*np.einsum('cdkl,klad->ca',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t2) + np.einsum('cdkl,klda->ca',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t2)
     
        I_oo += 2*np.einsum('cdkl,ilcd->ik',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.tau) - np.einsum('dckl,lidc->ik',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.tau) 
     
        Ioooo += np.einsum('cdkl,ijcd->ijkl',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t2)
     
        Iovvo += np.einsum('dclk,jlbd->jcbk',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t2) - np.einsum('dclk,jldb->jcbk',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t2) - 0.5*np.einsum('cdlk,jlbd->jcbk',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t2)
     
        Iovvo_2 += -0.5*np.einsum('dclk,jldb->jcbk',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t2)  - np.einsum('dckl,ljdb->jcbk',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t2)
      
        Iovov += -0.5*np.einsum('dckl,ildb->ickb',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t2)
        return I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov
     
        I_vv = None
        I_oo = None
        Iovvo = None
        Iovvo_2 = None
        Iovov = None
        gc.collect()
  
class amplitude(object):
    def __init__(self, data):

	self.nao = data.nao
	self.nocc = data.nocc
	self.nvirt = data.nvirt
	self.no_act = data.no_act

	self.data = data
	self.nv_act = data.nv_act
	self.n_act = self.no_act + self.nv_act

	self.data = data

    def doubles(self,I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2):
	occ = self.nocc
	nao = self.nao
        print " "
        R_ijab = 0.5*cp.deepcopy(self.data.twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
        R_ijab += -np.einsum('ik,kjab->ijab',I_oo,self.data.t2)        #diagrams linear 1 and non-linear 25,27,5,8,35,38'
        R_ijab += np.einsum('ca,ijcb->ijab',I_vv,self.data.t2)         #diagrams linear 2 and non-linear 24,26,34',6,7
        R_ijab += 0.5*np.einsum('cdab,ijcd->ijab',Ivvvv,self.data.tau) #diagrams linear 5 and non-linear 2
        R_ijab += 0.5*np.einsum('ijkl,klab->ijab',Ioooo,self.data.tau)  #diagrams linear 9 and non-linear 1,22,38
        R_ijab += 2*np.einsum('jcbk,kica->ijab',Iovvo,self.data.t2)    #diagrams linear 6 and non-linear 19,28,20
        R_ijab += - np.einsum('jcbk,ikca->ijab',Iovvo_2,self.data.t2)  #diagrams linear 8 and non-linear 21,29 
        R_ijab += - np.einsum('ickb,kjac->ijab',Iovov,self.data.t2)    #diagrams linear 10 and non-linear 23
        R_ijab += -np.einsum('icka,kjcb->ijab',Iovov_2,self.data.t2)   #diagram linear 7

        return R_ijab
     
        R_ijab = None
        I_oo = None
        I_vv = None
        Ivvvv = None
        Ioooo = None
        Iovvo = None
        Iovvo_2 = None
        Iovov = None
        Iovov_2 = None
        gc.collect()


