# iccsdn_modules

## Steps to pull the code from GitHub
1. Check if there exists the file ~/.ssh/id_rsa.pub . This file contains the public ssh-key which identifies your computer into the GitHub system. If the file does not exist, create it following the further steps:
   a) 'ssh-keygen -t rsa'
   b) Just press <Enter> to accept the default location and file name. If the .ssh directory doesn't exist, the system creates one for you.
   c) Press  <Enter> twice when you are prompted to enter and re-enter a passphrase. This means that you are not using a passphrase. 
2. Once you have the ssh-key, add it to your GitHub profile follwoing settings -> SSH and GPG keys -> New SSH key
3. Finally, use the command 'git clone git@github.com:PradiptaSamanta/iccsdn_modules.git' on the terminal of your computer. 

## Ways to run calculation using the modules present in this directory
1. The file 'calc.py' is the python script to run CC energy and excitation energy calculations. But a proper pythonpath needs to be set before running the calculations. This could be done with the command:
  'PYTHONPATH=/PATH/TO/THE/MODULES:$PYTHONPATH'
2.  
