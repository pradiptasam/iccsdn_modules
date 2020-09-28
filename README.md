# iccsdn_modules

## Steps to pull the code from GitHub
1. Check if there exists the file ~/.ssh/id_rsa.pub . This file contains the public ssh-key which identifies your computer into the GitHub system. If the file does not exist, create it following the further steps:

   a) 'ssh-keygen -t rsa'
   
   b) Just press <Enter> to accept the default location and file name. If the .ssh directory doesn't exist, the system creates one for you.
   
   c) Press  <Enter> twice when you are prompted to enter and re-enter a passphrase. This means that you are not using a passphrase. 
   
2. Once you have the ssh-key, add it to your GitHub profile follwoing settings -> SSH and GPG keys -> New SSH key

3. Finally, use the command 'git clone git@github.com:PradiptaSamanta/iccsdn_modules.git' on the terminal of your computer. 

## Ways to run calculation using the modules present in this directory

 'calc.py' is the python script with all the necessary commands written inside to run both the ground and excitation energy calculations. You can find comments and descriptions about most of the commands used in the script. This script, in principle, can be run successfully when it is still inside the directory, mainly because all the modules it calls are in the same directory or within the pyscf program. But, to run the same script from any other directory, you need to use either of these two alternative ways.   

I) The first is by augmenting the $PYTHONPATH which is a system environment containing all the directories to look for python scripts. The path to the directory 'iccsdn_modules' can be added to the $PYTHONPATH using the command:

  'PYTHONPATH=/PATH/TO/THE/MODULES:$PYTHONPATH'
  
II) The other would be by adding the directory 'iccsdn_modules' into the pyscf programs. i.e. inside the directory /PATH/TO/PYSCF/pyscf. In this case, instead of copying (or, moving) 'iccsdn_modules', it is always better to keep the directory inside pyscf as a soft link. You can do that following the command:

  'cd  /PATH/TO/PYSCF/pyscf/; ln -s /PATH/TO/ICCSDN_MODULES/ iccsdn'.
  
  Here I rename the soft link as 'iccsdn'. With this steps being done, you can now made all the changes you want inside the original 'iccsdn_modules' directory which will in turn get mirrored in the 'iccsdn' inside pyscf.
  
  You can choose either of these two ways to run the script. But depending on which one you choose, the command to import the CC and MP2 modules would differ. To make this import process automatic, you can add the following lines at the top of the 'calc.py' script.
  
  '''
  
import os 

_pythonpath = os.environ['PYTHONPATH']

_present_in_pythonpath =  _pythonpath.find('iccsdn_module') >= 0

if (_present_in_pythonpath):

  import CC
  
  import MP2
  
else:

  from pyscf.iccsdn import CC
  
  from pyscf.iccsdn import MP2
  
'''

You can also find these above mentioned command liness inside the scripts 'test/ccsd_excit.py' and 'test/iccsd_excit.py'. These two scripts along with their respective output files are there for debugging any future change in the module files.   
