from paramiko import SSHClient
import sys
import time

#Auxiliary function to run algogithms
def test_algs(ssh, dir, program_list):
    stdin, stdout, stderr = ssh.exec_command('ls -l; cd Project/NotImportantFiles/data-structures-algorithms-python-master/algorithms/'+str(dir)+'; ls')
    print(stdout.read())
    
    for p in program_list:
        stdin, stdout, stderr = ssh.exec_command('ls -l; cd Project/NotImportantFiles/data-structures-algorithms-python-master/algorithms/'+str(dir)+"; python3 "+str(p)+";")
        print(stdout.read())  



#Write a hello world file
def good_behaviour1(ssh):
    #Create the connection
    ssh.connect('192.168.1.1', username="labcom", password="labcom")
    channel = ssh.invoke_shell()
    
    channel.send("cd Project/\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    channel.send("ls -l\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    channel.send("cd NotImportantFiles/\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    channel.send("ls -l\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    channel.send("rm -rf helloworld.py\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    channel.send("ls -l\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    channel.send("touch helloworld.py\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    channel.send("ls -l\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))
 
    channel.send('echo "print(\'helloworld\')" >> helloworld.py\n')
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    channel.send("python3 helloworld.py\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    #Close the connection
    ssh.close()




#Execute a series of algorithmic tests
def good_behaviour2(ssh):
    #Create the connection
    ssh.connect('192.168.1.1', username="labcom", password="labcom")
    
    stdin, stdout, stderr = ssh.exec_command('ls -l; cd Project/; ls -l; cd NotImportantFiles/; ls -l')
    print(stdout.read())
    #Random Delay
    
    stdin, stdout, stderr = ssh.exec_command('ls -l; cd Project/NotImportantFiles/data-structures-algorithms-python-master/algorithms; ls -l')
    print(stdout.read())
    #Random Delay   
    
    
    test_algs(ssh, "1_BinarySearch", ["binary_search_exercise_solution.py", "binarysearch.py"])
    print("1 Done")
    test_algs(ssh, "2_BubbleSort", ["bubble_sort_exercise_solution.py", "bubble_sort.py"])
    print("2 Done")
    test_algs(ssh, "3_QuickSort", ["quick_sort_exercise_soluiton_lomuto.py", "quick_sort.py"])
    print("3 Done")
    test_algs(ssh, "4_InsertionSort", ["insertion_sort.py"])
    print("4 Done")
    test_algs(ssh, "5_MergeSort", ["merge_sort_exercise_solution.py", "merge_sort_primitive.py", "merge_sort_final.py"])
    print("5 Done")
    test_algs(ssh, "6_ShellSort", ["shell_sort.py", "shell_sort_exercise_solution.py"])
    print("6 Done")
    test_algs(ssh, "7_SelectionSort", ["selection_sort_exercise_solution.py", "selection_sort.py"])
    print("7 Done")
    test_algs(ssh, "8_DepthFirstSearch", ["dfs_exercise.py", "dfs.py"])
    print("8 Done")
    test_algs(ssh, "8_recursion", ["recursion.py"])
    print("9 Done")
    test_algs(ssh, "9_BreadthFirstSearch", ["bfs.py", "bfs_exercise_solution.py"])
    print("All Done")

    #Close the connection
    ssh.close()

#Execute a computational intensive program
def good_behaviour3(ssh):
    #Open Connection
    ssh.connect('192.168.1.1', username="labcom", password="labcom")
    
    
    stdin, stdout, stderr = ssh.exec_command('ls -l; cd Project/; ls -l; cd NotImportantFiles/; ls -l cd MD5/; ls -l;')
    print(stdout.read())
    #Random Delay 

    #PRINT AO FICHEIRO???
 
    stdin, stdout, stderr = ssh.exec_command('cd Project/; cd NotImportantFiles/MD5; python3 decriptmd5.py -f random.txt')
    print(stdout.read())
    #Random Delay

    #Close the connection
    ssh.close()



#Create SSH client
ssh = SSHClient()
ssh.load_system_host_keys()

good_behaviour1(ssh)
#good_behaviour2(ssh)
#good_behaviour3(ssh)
