import time
from subprocess import Popen, PIPE

#Function to run ssh commands
def run_ssh_cmd(psswd, host, cmd):
    cmds = ['sshpass', '-p', psswd, 'ssh', '-t', host, cmd]
    return Popen(cmds, stdout=PIPE, stderr=PIPE, stdin=PIPE)

#Function to run the scp command
def run_scp_cmd(psswd, host, filename, newfilename):
    cmds = ['sshpass', '-p', psswd, 'scp', str(host)+":"+filename, newfilename]
    return Popen(cmds, stdout=PIPE, stderr=PIPE, stdin=PIPE)

results = run_ssh_cmd('labcom', 'labcom@192.168.1.1', 'ls -l').stdout.read()
print(results)
results = run_scp_cmd('labcom', 'labcom@192.168.1.1', 'test.py', 'newCopiedTest.py')
print(results)
