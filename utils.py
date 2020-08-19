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
  
    def R_ia_intermediates(self):

	occ = self.nocc
	nao = self.nao
        I1 = 2*np.einsum('cbkj,kc->bj',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t1)
        I2 = -np.einsum('cbjk,kc->bj',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t1)
        return I1,I2
     
        I1 = None
        I2 = None
  
    def singles_intermediates(self,I_oo,I_vv,I2, rank_t1):

	occ = self.nocc
	nao = self.nao
	virt = self.nvirt

        I_oo += 2*np.einsum('ibkj,jb->ik',self.data.twoelecint_mo[:occ,occ:nao,:occ,:occ],self.data.t1)    #intermediate for diagrams 5
        I_oo += -np.einsum('ibjk,jb->ik',self.data.twoelecint_mo[:occ,occ:nao,:occ,:occ],self.data.t1)     #intermediate for diagrams 8

        I_vv += 2*np.einsum('bcja,jb->ca',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],self.data.t1)    #intermediate for diagrams 6
        I_vv += -np.einsum('cbja,jb->ca',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],self.data.t1)    #intermediate for diagrams 7
        if (rank_t1 > 1):
            I_vv += -2*np.einsum('dclk,ld,ka->ca',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t1,self.data.t1)  #intermediate for diagram 34'
        
        I_oovo = np.zeros((occ,occ,virt,occ))
        I_oovo += -np.einsum('cikl,jlca->ijak',self.data.twoelecint_mo[occ:nao,:occ,:occ,:occ],self.data.t2)    #intermediate for diagrams 11
        I_oovo += np.einsum('cdka,jicd->ijak',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],self.data.t2)    #intermediate for diagrams 12
        I_oovo += -np.einsum('jclk,lica->ijak',self.data.twoelecint_mo[:occ,occ:nao,:occ,:occ],self.data.t2)    #intermediate for diagrams 13
        I_oovo += 2*np.einsum('jckl,ilac->ijak',self.data.twoelecint_mo[:occ,occ:nao,:occ,:occ],self.data.t2)    #intermediate for diagrams 15
        I_oovo += -np.einsum('jckl,ilca->ijak',self.data.twoelecint_mo[:occ,occ:nao,:occ,:occ],self.data.t2)    #intermediate for diagrams 17
        
        I_vovv = np.zeros((virt,occ,virt,virt))
        I_vovv += np.einsum('cjkl,klab->cjab',self.data.twoelecint_mo[occ:nao,:occ,:occ,:occ],self.data.t2)    #intermediate for diagrams 9
        I_vovv += -np.einsum('cdlb,ljad->cjab',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],self.data.t2)    #intermediate for diagrams 10
        I_vovv += -np.einsum('cdka,kjdb->cjab',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],self.data.t2)    #intermediate for diagrams 14
        I_vovv += 2*np.einsum('cdal,ljdb->cjab',self.data.twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],self.data.t2)    #intermediate for diagrams 16
        I_vovv += -np.einsum('cdal,jldb->cjab',self.data.twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],self.data.t2)    #intermediate for diagrams 18
     
        I_oooo_2 = np.zeros((occ,occ,occ,occ))
	if (rank_t1 > 1):
            Ioooo_2 = 0.5*np.einsum('cdkl,ic,jd->ijkl',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t1,self.data.t1)    #intermediate for diagrams 37

        I_voov = -np.einsum('cdkl,kjdb->cjlb',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t2)    #intermediate for diagrams 39
     
        Iovov_3 = -np.einsum('dckl,ildb->ickb',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t2)  #intermediate for diagrams 36
        
        Iovvo_3 = 2*np.einsum('dclk,jlbd->jcbk',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t2) 
        Iovvo_3 += - np.einsum('dclk,jldb->jcbk',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t2) 
        Iovvo_3 += np.einsum('cdak,ic->idak',self.data.twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],self.data.t1)  #intermediate for diagrams 32,33,31
        Iovvo_3 += -np.einsum('iclk,la->icak',self.data.twoelecint_mo[:occ,occ:nao,:occ,:occ],self.data.t1)  #intermediate for diagram 30
        
        
        Iooov = np.einsum('dl,ijdb->ijlb',I2,self.data.t2) #intermediate for diagram 34
     
        I3 = np.zeros((occ,virt,virt,occ))
	if (rank_t1 > 1):
            I3 = -np.einsum('cdkl,ic,ka->idal',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,:occ],self.data.t1,self.data.t1)  #intermediate for diagram 40
        return I_oo, I_vv, I_oovo, I_vovv, Ioooo_2, I_voov, Iovov_3, Iovvo_3, Iooov, I3
     
        I_vv = None
        I_oo = None
        I_oovo = None
        I_vovv = None
        Ioooo_2 = None
        I_voov = None
        Iovov_3 = None
        Iovvo_3 = None
        Iooov = None
        I3 = None
        gc.collect()

    def W1_int_So(self):

        occ = self.nocc
        nao = self.nao
        o_act = self.no_act
       
        II_oo = np.zeros((occ,occ)) 
        II_oo[:,occ-o_act:occ] += -2*0.25*np.einsum('ciml,mlcv->iv',self.data.twoelecint_mo[occ:nao,:occ,:occ,:occ],self.data.So) 
        II_oo[:,occ-o_act:occ] += 0.25*np.einsum('diml,lmdv->iv',self.data.twoelecint_mo[occ:nao,:occ,:occ,:occ],self.data.So)
       
        return II_oo
        gc.collect()
    
    def W1_int_Sv(self):

        occ = self.nocc
        nao = self.nao
        virt = self.nvirt
        v_act = self.nv_act
       
        II_vv = np.zeros((virt,virt))
        II_vv[:v_act,:] += 2*0.25*np.einsum('dema,mude->ua',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],self.data.Sv) 
        II_vv[:v_act,:] += - 0.25*np.einsum('dema,mued->ua',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],self.data.Sv)
        return II_vv
        gc.collect()

    def coupling_terms_Sv(self):

        occ = self.nocc
        nao = self.nao
        virt = self.nvirt
        o_act = self.no_act
        v_act = self.nv_act
       
        II_vo = np.zeros((virt,o_act))
        II_vo[:v_act,:] += 2*0.25*np.einsum('cblv,lwcb->wv',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,occ-o_act:occ],self.data.Sv) 
        II_vo[:v_act,:] += - 0.25*np.einsum('bclv,lwcb->wv',self.data.twoelecint_mo[occ:nao,occ:nao,:occ,occ-o_act:occ],self.data.Sv)
      
        return II_vo
        gc.collect()
    
    def coupling_terms_So(self):

        occ = self.nocc
        nao = self.nao
        virt = self.nvirt
        o_act = self.no_act
        v_act = self.nv_act
       
        II_ov = np.zeros((v_act,occ)) 
        II_ov[:,occ-o_act:occ] += -2*0.25*np.einsum('dulk,lkdx->ux',self.data.twoelecint_mo[occ:nao,occ:occ+v_act,:occ,:occ],self.data.So)
        II_ov[:,occ-o_act:occ] +=  0.25*np.einsum('dulk,kldx->ux',self.data.twoelecint_mo[occ:nao,occ:occ+v_act,:occ,:occ],self.data.So) 
        
        return II_ov
        gc.collect()

    ##Two body Intermediates for (V_S_T)_c terms contributing to R_iuab and R_ijav##
    def W2_int_So(self):

        occ = self.nocc
        nao = self.nao
        virt = self.nvirt
        o_act = self.no_act
        v_act = self.nv_act
       
        II_ovoo = np.zeros((occ,virt,o_act,occ))
        II_ovoo3 = np.zeros((occ,v_act,occ,occ))
        II_vvvo3 = np.zeros((virt,v_act,virt,occ))
     
        II_ovoo[:,:,:,occ-o_act:occ] += -np.einsum('cdvk,jkcw->jdvw',self.data.twoelecint_mo[occ:nao,occ:nao,occ-o_act:occ,:occ],self.data.So)
     
        ##Intermediates for off diagonal terms like So->R_iuab##
     
        II_ovoo3[:,:,:,occ-o_act:occ] += -np.einsum('dulk,ikdw->iulw',self.data.twoelecint_mo[occ:nao,occ:occ+v_act,:occ,:occ],self.data.So)
        II_vvvo3[:,:,:,occ-o_act:occ] += -np.einsum('dulk,lkaw->duaw',self.data.twoelecint_mo[occ:nao,occ:occ+v_act,:occ,:occ],self.data.So)
      
        return II_ovoo,II_ovoo3,II_vvvo3
        gc.collect()
    
    
    def W2_int_Sv(self):

        occ = self.nocc
        nao = self.nao
        virt = self.nvirt
        o_act = self.no_act
        v_act = self.nv_act

        II_vvvo = np.zeros((v_act,virt,virt,occ))
        II_vvvo2 = np.zeros((virt,virt,virt,o_act))
        II_ovoo2 = np.zeros((occ,virt,occ,o_act))
      
        II_vvvo[:,:v_act,:,:] += -np.einsum('uckl,kxbc->uxbl', self.data.twoelecint_mo[occ:occ+v_act,occ:nao,:occ,:occ],self.data.Sv) 
        ##Intermediates for off diagonal terms like Sv->R_ijav##
        II_vvvo2[:,:v_act,:,:] += -np.einsum('dckv,kxac->dxav', self.data.twoelecint_mo[occ:nao,occ:nao,:occ,occ-o_act:occ],self.data.Sv)
        II_ovoo2[:,:v_act,:,:] += np.einsum('dckv,ixdc->ixkv', self.data.twoelecint_mo[occ:nao,occ:nao,:occ,occ-o_act:occ],self.data.Sv)
     
        return II_vvvo,II_vvvo2,II_ovoo2
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

    def singles(self,I1,I2,I_oo,I_vv):

	occ = self.nocc
	nao = self.nao
        R_ia = cp.deepcopy(self.data.fock_mo[:occ,occ:nao])
        R_ia += -np.einsum('ik,ka->ia',I_oo,self.data.t1)                                          #diagrams 1,l,j,m,n
        R_ia += np.einsum('ca,ic->ia',I_vv,self.data.t1)                                           #diagrams 2,k,i
        R_ia += -2*np.einsum('ibkj,kjab->ia',self.data.twoelecint_mo[:occ,occ:nao,:occ,:occ],self.data.tau)     #diagrams 5 and a
        R_ia += np.einsum('ibkj,jkab->ia',self.data.twoelecint_mo[:occ,occ:nao,:occ,:occ],self.data.tau)     #diagrams 6 and b
        R_ia += 2*np.einsum('cdak,ikcd->ia',self.data.twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],self.data.tau) #diagrams 7 and c
        R_ia += -np.einsum('cdak,ikdc->ia',self.data.twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],self.data.tau) #diagrams 8 and d
        R_ia += 2*np.einsum('bj,ijab->ia',I1,self.data.t2) - np.einsum('bj,ijba->ia',I1,self.data.t2)     #diagrams e,f
        R_ia += 2*np.einsum('bj,ijab->ia',I2,self.data.t2) - np.einsum('bj,ijba->ia',I2,self.data.t2)     #diagrams g,h
        R_ia += 2*np.einsum('icak,kc->ia',self.data.twoelecint_mo[:occ,occ:nao,occ:nao,:occ],self.data.t1)           #diagram 3
        R_ia += -np.einsum('icka,kc->ia',self.data.twoelecint_mo[:occ,occ:nao,:occ,occ:nao],self.data.t1)           #diagram 4
        return R_ia
     
        R_ia = None
        I_oo = None
        I_vv = None
        I1 = None
        I2 = None
        gc.collect()

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


    def singles_n_doubles(self, I_oovo,I_vovv, rank_t1):

	occ = self.nocc
	nao = self.nao
        R_ijab = -np.einsum('ijak,kb->ijab',I_oovo,self.data.t1)       #diagrams 11,12,13,15,17
        R_ijab += np.einsum('cjab,ic->ijab',I_vovv,self.data.t1)       #diagrams 9,10,14,16,18
        R_ijab += -np.einsum('ijkb,ka->ijab',self.data.twoelecint_mo[:occ,:occ,:occ,occ:nao],self.data.t1)            #diagram 3
        R_ijab += np.einsum('cjab,ic->ijab',self.data.twoelecint_mo[occ:nao,:occ,occ:nao,occ:nao],self.data.t1)       #diagram 4

	if (rank_t1 > 1):
            R_ijab += -np.einsum('ickb,ka,jc->ijab',self.data.twoelecint_mo[:occ,occ:nao,:occ,occ:nao],self.data.t1,self.data.t1)   #diagrams non-linear 3
            R_ijab += -np.einsum('icak,jc,kb->ijab',self.data.twoelecint_mo[:occ,occ:nao,occ:nao,:occ],self.data.t1,self.data.t1)   #diagrams non-linear 4
        return R_ijab
     
        R_ijab = None
        I_oovo = None
        I_vovv = None
        gc.collect() 

    def higher_order(self, Iovov_3,Iovvo_3,Iooov,I3,Ioooo_2,I_voov, rank_t1):

        R_ijab = -np.einsum('ickb,jc,ka->ijab',Iovov_3,self.data.t1,self.data.t1)       #diagrams 36
        R_ijab += -np.einsum('jcbk,ic,ka->ijab',Iovvo_3,self.data.t1,self.data.t1)      #diagrams 32,33,31,30
        R_ijab += -np.einsum('ijlb,la->ijab',Iooov,self.data.t1)      #diagram 34,30
        R_ijab += -0.5*np.einsum('idal,jd,lb->ijab',I3,self.data.t1,self.data.t1)      #diagram 40
        R_ijab += np.einsum('ijkl,klab->ijab',Ioooo_2,self.data.t2)      #diagram 37
        R_ijab += -np.einsum('cjlb,ic,la->ijab',I_voov,self.data.t1,self.data.t1)      #diagram 39
        return R_ijab
     
        R_ijab = None 
        Iovov_3 = None 
        Iovvo_3 = None 
        Iooov = None 
        I3 = None 
        Ioooo_2 = None 
        I_voov = None
        gc.collect()
    
    def inserted_diag_So_t1(self, II_oo):

        R_ia = -np.einsum('ik,ka->ia',II_oo, self.data.t1)
        return R_ia 
     
        R_ia = None
        II_oo = None
        gc.collect() 
    
    def inserted_diag_Sv_t1(self, II_vv):

        R_ia = np.einsum('ca,ic->ia',II_vv, self.data.t1)
        return R_ia
     
        R_ia = None
        II_vv = None
        gc.collect() 

    def inserted_diag_So(self, II_oo):

        R_ijab = -np.einsum('ik,kjab->ijab',II_oo,self.data.t2)   
        return R_ijab 
     
        R_ijab = None
        II_oo = None
        gc.collect() 
    
    def inserted_diag_Sv(self, II_vv):

        R_ijab = np.einsum('ca,ijcb->ijab',II_vv,self.data.t2)  
        return R_ijab
     
        R_ijab = None
        II_vv = None
        gc.collect() 

    def Sv_diagram_vs_contraction(self):

        occ = self.nocc
        nao = self.nao
        virt = self.nvirt
        o_act = self.no_act
        v_act = self.nv_act

        R_iuab = cp.deepcopy(self.data.twoelecint_mo[:occ,occ:occ+v_act,occ:nao,occ:nao])
        R_iuab += -np.einsum('ik,kuab->iuab',self.data.fock_mo[:occ,:occ],self.data.Sv)
        R_iuab += np.einsum('da,iudb->iuab',self.data.fock_mo[occ:nao,occ:nao],self.data.Sv)
        R_iuab += np.einsum('db,iuad->iuab',self.data.fock_mo[occ:nao,occ:nao],self.data.Sv)
        R_iuab += np.einsum('edab,iued->iuab',self.data.twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],self.data.Sv)
        R_iuab += 2*np.einsum('idak,kudb->iuab',self.data.twoelecint_mo[:occ,occ:nao,occ:nao,:occ],self.data.Sv) 
        R_iuab += -np.einsum('idka,kudb->iuab',self.data.twoelecint_mo[:occ,occ:nao,:occ,occ:nao],self.data.Sv)
        R_iuab += -np.einsum('dika,kubd->iuab',self.data.twoelecint_mo[occ:nao,:occ,:occ,occ:nao],self.data.Sv)
        R_iuab += -np.einsum('idkb,kuad->iuab',self.data.twoelecint_mo[:occ,occ:nao,:occ,occ:nao],self.data.Sv)
        return R_iuab
        
        R_iuab = None
        gc.collect()
    
    def Sv_diagram_vt_contraction(self):

        occ = self.nocc
        nao = self.nao
        o_act = self.no_act
        v_act = self.nv_act

        R_iuab = 2*np.einsum('dukb,kida->iuab',self.data.twoelecint_mo[occ:nao,occ:occ+v_act,:occ,occ:nao],self.data.t2)
        R_iuab += -np.einsum('udkb,kida->iuab',self.data.twoelecint_mo[occ:occ+v_act,occ:nao,:occ,occ:nao],self.data.t2)
        R_iuab += -np.einsum('dukb,kiad->iuab',self.data.twoelecint_mo[occ:nao,occ:occ+v_act,:occ,occ:nao],self.data.t2)
        R_iuab += np.einsum('uikl,klba->iuab', self.data.twoelecint_mo[occ:occ+v_act,:occ,:occ,:occ],self.data.t2)
        R_iuab += -np.einsum('udka,kibd->iuab',self.data.twoelecint_mo[occ:occ+v_act,occ:nao,:occ,occ:nao],self.data.t2)
        return R_iuab
        
        R_iuab = None
        gc.collect() 
     
    def So_diagram_vs_contraction(self):

        occ = self.nocc
        nao = self.nao
        o_act = self.no_act
        v_act = self.nv_act

        R_ijav = cp.deepcopy(self.data.twoelecint_mo[:occ,:occ,occ:nao,occ-o_act:occ])
        R_ijav += np.einsum('da,ijdv->ijav', self.data.fock_mo[occ:nao,occ:nao],self.data.So)
        R_ijav += -np.einsum('jl,ilav->ijav',self.data.fock_mo[:occ,:occ],self.data.So)
        R_ijav += -np.einsum('il,ljav->ijav',self.data.fock_mo[:occ,:occ],self.data.So)
        R_ijav += 2*np.einsum('dila,ljdv->ijav',self.data.twoelecint_mo[occ:nao,:occ,:occ,occ:nao],self.data.So)
        R_ijav += -np.einsum('dila,jldv->ijav', self.data.twoelecint_mo[occ:nao,:occ,:occ,occ:nao],self.data.So)   
        R_ijav += -np.einsum('dial,ljdv->ijav', self.data.twoelecint_mo[occ:nao,:occ,occ:nao,:occ],self.data.So)
        R_ijav += np.einsum('ijlm,lmav->ijav',  self.data.twoelecint_mo[:occ,:occ,:occ,:occ],self.data.So)
        R_ijav += -np.einsum('jdla,ildv->ijav', self.data.twoelecint_mo[:occ,occ:nao,:occ,occ:nao],self.data.So)
        return R_ijav
        
        R_ijav = None
        gc.collect()
    
    def So_diagram_vt_contraction(self):

        occ = self.nocc
        nao = self.nao
        o_act = self.no_act
        v_act = self.nv_act

        R_ijav = -np.einsum('djlv,liad->ijav',  self.data.twoelecint_mo[occ:nao,:occ,:occ,occ-o_act:occ],self.data.t2)
        R_ijav += -np.einsum('djvl,lida->ijav', self.data.twoelecint_mo[occ:nao,:occ,occ-o_act:occ,:occ],self.data.t2)
        R_ijav += np.einsum('cdva,jicd->ijav',  self.data.twoelecint_mo[occ:nao,occ:nao,occ-o_act:occ,occ:nao],self.data.t2)
        R_ijav += -np.einsum('idlv,ljad->ijav', self.data.twoelecint_mo[:occ,occ:nao,:occ,occ-o_act:occ],self.data.t2)
        R_ijav += 2*np.einsum('djlv,lida->ijav',self.data.twoelecint_mo[occ:nao,:occ,:occ,occ-o_act:occ],self.data.t2)
        return R_ijav
     
        R_ijav = None
        gc.collect()
    
    def T1_contribution_Sv(self):

        occ = self.nocc
        nao = self.nao
        o_act = self.no_act
        v_act = self.nv_act

        R_iuab = -np.einsum('uika,kb->iuab', self.data.twoelecint_mo[occ:occ+v_act,:occ,:occ,occ:nao],self.data.t1)
        R_iuab += np.einsum('duab,id->iuab', self.data.twoelecint_mo[occ:nao,occ:occ+v_act,occ:nao,occ:nao],self.data.t1)
        R_iuab += -np.einsum('iukb,ka->iuab',self.data.twoelecint_mo[:occ,occ:occ+v_act,:occ,occ:nao],self.data.t1)
        return R_iuab
     
        R_iuab = None
        gc.collect()
    
    def T1_contribution_So(self):

        occ = self.nocc
        nao = self.nao
        o_act = self.no_act
        v_act = self.nv_act

        R_ijav = np.einsum('diva,jd->ijav',  self.data.twoelecint_mo[occ:nao,:occ,occ-o_act:occ,occ:nao],self.data.t1)
        R_ijav += np.einsum('djav,id->ijav', self.data.twoelecint_mo[occ:nao,:occ,occ:nao,occ-o_act:occ],self.data.t1)
        R_ijav += -np.einsum('ijkv,ka->ijav',self.data.twoelecint_mo[:occ,:occ,:occ,occ-o_act:occ],self.data.t1)
        return R_ijav
     
        R_ijav = None
        gc.collect()

    def v_so_t_contraction_diag(self, II_ov):

        R_iuab = -np.einsum('ux,xiba->iuab',II_ov,self.data.t2)
        return R_iuab
     
        R_iuab = None
        II_ov = None
        gc.collect()
    
    def v_sv_t_contraction_diag(self, II_vo):

        R_ijav = np.einsum('wv,jiwa->ijav',II_vo,self.data.t2)
        return R_ijav
     
        R_ijav = None
        II_vo = None
        gc.collect()

    def w2_diag_So(self, II_ovoo,II_vvvo2,II_ovoo2):

        R_ijav = 2.0*np.einsum('jdvw,wida->ijav',II_ovoo,self.data.t2)
        R_ijav += -np.einsum('jdvw,wiad->ijav',II_ovoo,self.data.t2) #diagonal terms
        R_ijav += np.einsum('dxav,ijdx->ijav',II_vvvo2,self.data.t2) #off-diagonal terms
        R_ijav += -np.einsum('ixkv,kjax->ijav',II_ovoo2,self.data.t2)
        R_ijav += -np.einsum('jxkv,kixa->ijav',II_ovoo2,self.data.t2)
        return R_ijav
     
        R_ijav = None
        II_ovoo = None
        II_vvvo2 = None
        II_ovoo2 = None
        gc.collect()
    
    def w2_diag_Sv(self, II_vvvo,II_ovoo3,II_vvvo3):

        R_iuab = 2.0*np.einsum('uxbl,ilax->iuab',II_vvvo,self.data.t2)
        R_iuab += -np.einsum('uxbl,ilxa->iuab',II_vvvo,self.data.t2)
        R_iuab += -np.einsum('iulw,lwab->iuab',II_ovoo3,self.data.t2)
        R_iuab += -np.einsum('duaw,iwdb->iuab',II_vvvo3,self.data.t2) 
        R_iuab += -np.einsum('dubw,iwad->iuab',II_vvvo3,self.data.t2) 
        return R_iuab
     
        R_iuab = None
        II_vvvo = None
        II_ovoo3 = None
        II_vvvo3 = None
        gc.collect()
    
