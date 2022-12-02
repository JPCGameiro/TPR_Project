import spur
import time
from auxFunctions import *


#Create a python file and put hello world code inside it
def good_behaviour1():
    #Create the connection
    shell = spur.SshShell(hostname="192.168.1.1", username="labcom", password="labcom")
    
    #delay random
    shell.run(["ls","-l"])
    #delay random
    shell.run(["ls", "Project/"])
    #delay random
    shell.run(["cd", "Project/"])
    #delay random
    shell.run(["cd", "NotImportantFiles/"])
    #delay random
    shell.run(["ls", "-l"])
    #delay random
    shell.run(["rm", "-rf", "helloworld.py"])
    #delay random
    shell.run(["touch", "helloworld.py"])
    #delay random
    shell.run(["echo",">>", "print('hello world')", "helloworld.py"])
    #delay random
    shell.run(["cat", "helloworld.py"])
    #delay random
    shell.run(["python3", "helloworld.py"])
    #delay random
    shell.run(["echo", ">", "",""])
    #delay random
    shell.run(["cat", "helloworld.py"])
    #delay random
    shell.run(["python3 helloworld.py"])
    #delay random
    shell.run(["cd", ".."])
    #delay random
    shell.run(["cd", ".."])
    #delay random
    
    #Close the connection
    shell.close()



#Run algorithm tests
def good_behaviour2(shell):
    #Create the connection
    shell = spur.SshShell(hostname="192.168.1.1", username="labcom", password="labcom")

    shell.run(["cd", "Project"])
    shell.run(["cd", "NotImportantFiles"])
    shell.run(["cd", "data-structures-algorithms-python-master"])
    shell.run(["cd", "algorithms"])
    
    test_algs1(shell)
    test_algs2(shell)
    test_algs3(shell)
    test_algs4(shell)
    test_algs5(shell)
    test_algs6(shell)
    test_algs7(shell)
    test_algs8(shell)
    test_algs9(shell)
    test_algs10(shell)

    shell.run(["cd", "../../../.."])

    #Close the connection
    shell.close()



def good_behavior3():
    #Create the connection
    shell = spur.SshShell(hostname="192.168.1.1", username="labcom", password="labcom")

    shell.run(["ip", "a"])
    shell.run(["lscpu"])
    shell.run(["ps", "aux"])

    #Close the connection
    shell.close()





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

