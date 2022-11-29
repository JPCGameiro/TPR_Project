import spur
import time

#Create the connection
shell = spur.SshShell(hostname="192.168.1.1", username="labcom", password="labcom")

#Execute commands
with shell:
    result = shell.run(["echo", "-n", "hello"])
    time.sleep(2)
    print(result.output)
    result = shell.run(["ls", "-l"])
    time.sleep(2)
    print(result.output)
    result = shell.run(["echo", "test"])
    time.sleep(2)

#Close the connection
shell.close()

print(result.output)
