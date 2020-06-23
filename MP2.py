import numpy as np

from pyscf import scf, mp
from pyscf.lib import logger

from PostHF import GetIntNData

class MP2(object):
    def __init__(self, mf, nfo=0, nfv=0):

        self.mol  = mf.mol
        self.mf = mf
#       self.verbose = self.mol.verbose
#       self.stdout = self.mol.stdout

        self.nfo = nfo
        self.nfv = nfv

	self.nel = self.mol.nelectron

	self.e_hf = mf.e_tot

    def check_mp2(self, e_mp2):
        m = mp.MP2(self.mf)
        if abs(m.kernel()[0]-e_mp2) <= 1E-6:
            print "MP2 successfully done"
	else:
	    print "MP2 energies differ"

    def calc_mp2_energy(self):

        e_mp2 = 2*np.einsum('ijab,ijab',self.t2,self.twoelecint_mo[:self.nocc,:self.nocc,self.nocc:self.nao,self.nocc:self.nao]) - np.einsum('ijab,ijba',self.t2,self.twoelecint_mo[:self.nocc,:self.nocc,self.nocc:self.nao,self.nocc:self.nao])
        print "MP2 correlation energy is : "+str(e_mp2)

        e_mp2_tot = self.e_hf + e_mp2
        print "MP2 energy is : "+str(e_mp2_tot)
	
	self.check_mp2(e_mp2)

    def run(self):

	#log = logger.Logger(self.stdout, self.verbose)
	#log.warn('**** RUNING MP2 ****')

	AllData = GetIntNData(self.mf, self.nfo, self.nfv)

	AllData.transform_all_ints()

	self.twoelecint_mo = AllData.twoelecint_mo

	self.nao = AllData.nao
	self.nocc = AllData.nocc
	self.nvirt = AllData.nvirt

	AllData.init_guess_t2()
	self.t2 = AllData.t2

	self.calc_mp2_energy()

        print '**** RUNING MP2 ****'

