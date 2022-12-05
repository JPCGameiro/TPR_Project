from paramiko import SSHClient
from scp import SCPClient
import sys

# Define progress callback that prints the current percentage completed for the file
def progress(filename, size, sent):
    sys.stdout.write("%s's progress: %.2f%%   \r" % (filename, float(sent)/float(size)*100) )



#Create SSH client
ssh = SSHClient()
ssh.load_system_host_keys()

#Split file in chunks
ssh.connect('192.168.1.1', username="labcom", password="labcom")
stdin, stdout, stderr = ssh.exec_command('cd Project/ImportantFiles; ls -l; split -b 256M 1GB.zip newfile; ls -l')
s = str(stdout.read())
print(s)
ssh.close()

numFiles = s.count("newfile")
#Random Delay


#Copy chunks
for i in range(0, numFiles):
    #Open ssh connection
    ssh.connect('192.168.1.1', username="labcom", password="labcom")

    # SCPCLient takes a paramiko transport as an argument and progress callback to see status of the file
    scp = SCPClient(ssh.get_transport(), progress=progress)

    #Copy file from remote server
    scp.get('/home/labcom/Project/ImportantFiles/newfilea'+chr(97+i))

    #Close the scp connection
    scp.close()

    #Close the ssh connection
    ssh.close()

    #Random Delay - TEMOS DE FALAR SOBRE ESTE


#Delete traces of the chunks
ssh.connect('192.168.1.1', username="labcom", password="labcom")
stdin, stdout, stderr = ssh.exec_command('cd Project/ImportantFiles; rm -rf newfile*; ls -l')
s = str(stdout.read())
print(s)
ssh.close()