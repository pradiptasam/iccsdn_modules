import pyscf
from pyscf import gto, scf
import MP2
import CC

##---------------------------------------------------------##
             #Specify geometry and basis set#       
##---------------------------------------------------------##

# The system and basis set used in the calculation are defined using the pyscf.gto class 
mol = pyscf.gto.M(
verbose = 5,
output = None,
unit='Bohr',
atom ='''
Li  0.000000,  0.000000, -0.3797714041
H   0.000000,  0.000000,  2.6437904102
''',
basis = 'sto-3g',
symmetry = 'C2v',
)

##---------------------------------------------------------##

#The 'scf.RHF' class from the pyscf program is called here
mf = scf.RHF(mol).run()

# If you wish, call the MP2.MP2 class from within the iccsdn_modules
#mp2_res = MP2.MP2(mf)
#mp2_res.nfo = 1
#mp2_res.run()

# cc_res is linked to the class CC.state
cc_res = CC.state(mf)

# The default of CC.state.variant is CCSD, so one can run CCSD without adding the line mentioning the variant.
# An alternative way to do 'ICCSD' is: 'cc_res = CC.state(mf, variant='ICCSD')
# The name of the method can be both in upper and lower cases, and it can also contain '-'. So, to do iCCSDn-PT, all of these - 'iccsd-pt', 'iCCSDn-PT', 'ICCSDPT' - are acceptable.
cc_res.variant = 'CCSD'

# These are also the default values for no_act and nv_act. Soi, there is no need to have these lines unless you 
#are using different values for no_act and nv_act
#if (cc_res.variant == 'ICCSD'):
#    cc_res.no_act = 1
#    cc_res.nv_act = 1

# maxsub represents the maximum dimension of the subspace being used for the DIIS method. The same parameter is
# also used as the maximum dimension of the subspace while using the Davidson method in the excitation energy calculation.
# The default value for maxsub is 20
cc_res.maxsub = 30

# Maximum number of iterations to be used to solve the amplitude equation. Again, same parameter is used in the similar 
#context while calculating excitation energy.
# The default value for maxiter is 50
cc_res.maxiter = 30

# Convergence threshold used to solve the amplitude. Ditto for excitation energy calculation.
# The default value for conv is 1e-7
cc_res.conv = 1e-7

# The class 'CC.state.energy', and therefore 'cc_res.energy' contains all the routines to solve the amplitude for CC methods 
# and to calculate the final energy. Among them, 'run' is the main routine to be called.
cc_res.energy.run()

# Change the next three parameters only if you want to reassign them to new values for the excitation energy calculation.
cc_res.maxiter = 30
cc_res.conv = 1e-6
cc_res.maxsub=6

# The class 'CC.state.exc_en', and therefore 'cc_res.exc_en' contains all the routines and parameters used to calculate excitation energy. 
# The most important input parameter here is 'root_info', which is used in order to mention the number of excited states for each symmetry 
# to be calculated. The default is [1]
cc_res.exc_en.root_info = [1,0,0,0]

# The following parameter is liitle bit insignificant at this point. The default used for this is False.
#cc_res.exc_en.tUseOtherRoots=True

# The main subroutine that needs to be called to get all the excitation energies.
cc_res.exc_en.run()
