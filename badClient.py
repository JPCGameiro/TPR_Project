from paramiko import SSHClient
from scp import SCPClient

ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect('192.168.1.1', username="labcom", password="labcom")

# SCPCLient takes a paramiko transport as an argument
scp = SCPClient(ssh.get_transport())

#Put file on remote server
scp.put('test.txt', 'test2.txt')

#Copy file from remote server
scp.get('test2.txt')

# Uploading the 'test' directory with its content in the
# '/home/labcom' remote directory
scp.put('test', recursive=True, remote_path='/home/labcom')

scp.close()