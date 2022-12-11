from paramiko import SSHClient
import sys
import time
from random import randrange, random

#random.gauss(mean, deviation)

#Auxiliary function to run algogithms
def test_algs(channel, dir, program_list):
    channel.send("cd "+str(dir)+"\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))
    time.sleep(abs(random.gauss(2,1)))                    #+- 0-3s


    for p in program_list:
        channel.send("python3 "+str(p)+"\n")
        while not channel.recv_ready():
            time.sleep(0.5)
        out = channel.recv(9999)
        print(out.decode("ascii"))
        time.sleep(abs(random.gauss(10,5)))                #+- 0-20s corre um programa, ta lá a olhar pros resultados e corre a seguir
    
    channel.send("cd ..\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))







#Write a hello world file
def good_behaviour1(ssh):
    #Create the connection
    ssh.connect('192.168.1.1', username="labcom", password="labcom")
    channel = ssh.invoke_shell()
    
    time.sleep(abs(random.gauss(2,1)))                    #+- 0-3s

    channel.send("cd Project/\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(1,1)))                    #+- 0-2s

    channel.send("ls -l\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(1,1)))                    #+- 0-2s

    channel.send("cd NotImportantFiles/\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(1,1)))                    #+- 0-2s

    channel.send("ls -l\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(2,1)))                    #+- 0-3s

    channel.send("rm -rf helloworld.py\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(1,1)))                    #+- 0-2s

    channel.send("ls -l\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(3,1)))                    #+- 1-4s

    channel.send("touch helloworld.py\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(1,1)))                    #+- 0-2s

    channel.send("ls -l\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(6,4)))                    #+- 0-11s ta a pensar no que escrever
 
    channel.send('echo "print(\'helloworld\')" >> helloworld.py\n')
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(1,1)))                    #+- 0-2s

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
    channel = ssh.invoke_shell()

    time.sleep(abs(random.gauss(2,1)))                    #+- 0-3s
    
    channel.send("ls -l\n ")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))
   
    time.sleep(abs(random.gauss(1,1)))                    #+- 0-2s
    
    channel.send("cd Project/NotImportantFiles/\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))  

    time.sleep(abs(random.gauss(1,1)))                    #+- 0-2s
    
    channel.send("cd  data-structures-algorithms-python-master/algorithms\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(1,1)))                    #+- 0-2s

    channel.send("ls -l\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))
    
    
    
    i = randrange(10)
    if (i <= 3):
        test_algs(channel, "1_BinarySearch", ["binary_search_exercise_solution.py", "binarysearch.py"])
        #Random delay
        time.sleep(abs(random.gauss(5,3)))      #+- 0-8s pq é tipo onde ele escolhe o cd para o proximo algoritmo
        test_algs(channel, "2_BubbleSort", ["bubble_sort_exercise_solution.py", "bubble_sort.py"])
        #Random delay
        time.sleep(abs(random.gauss(5,3)))      #+- 0-8s
        test_algs(channel, "3_QuickSort", ["quick_sort_exercise_soluiton_lomuto.py", "quick_sort.py"])
    elif ( i > 3 and i < 8 ):
        test_algs(channel, "4_InsertionSort", ["insertion_sort.py"])
        #Random delay
        time.sleep(abs(random.gauss(5,3)))      #+- 0-8s
        test_algs(channel, "5_MergeSort", ["merge_sort_exercise_solution.py", "merge_sort_primitive.py", "merge_sort_final.py"])
        #Random delay
        time.sleep(abs(random.gauss(5,3)))      #+- 0-8s
        test_algs(channel, "6_ShellSort", ["shell_sort.py", "shell_sort_exercise_solution.py"])
        #Random delay
        time.sleep(abs(random.gauss(5,3)))      #+- 0-8s
        test_algs(channel, "7_SelectionSort", ["selection_sort_exercise_solution.py", "selection_sort.py"])
    else:
        test_algs(channel, "8_DepthFirstSearch", ["dfs_exercise.py", "dfs.py"])
        #Random delay
        time.sleep(abs(random.gauss(5,3)))      #+- 0-8s
        test_algs(channel, "8_recursion", ["recursion.py"])
        #Random delay
        time.sleep(abs(random.gauss(5,3)))      #+- 0-8s
        test_algs(channel, "9_BreadthFirstSearch", ["bfs.py", "bfs_exercise_solution.py"])

    time.sleep(abs(random.gauss(7,4)))      #+- 4-12s

    i = random.uniform(0, 1)
    while (i < 0.5):
        i = randrange(10)
        if (i == 0):
            test_algs(channel, "1_BinarySearch", ["binary_search_exercise_solution.py", "binarysearch.py"])
        elif (i==1):
            test_algs(channel, "2_BubbleSort", ["bubble_sort_exercise_solution.py", "bubble_sort.py"])
        elif (i==2):
            test_algs(channel, "3_QuickSort", ["quick_sort_exercise_soluiton_lomuto.py", "quick_sort.py"])
        elif (i==3):
            test_algs(channel, "4_InsertionSort", ["insertion_sort.py"])
        elif (i==4):
            test_algs(channel, "5_MergeSort", ["merge_sort_exercise_solution.py", "merge_sort_primitive.py", "merge_sort_final.py"])
        elif (i==5):
            test_algs(channel, "6_ShellSort", ["shell_sort.py", "shell_sort_exercise_solution.py"])
        elif (i==6):
            test_algs(channel, "7_SelectionSort", ["selection_sort_exercise_solution.py", "selection_sort.py"])
        elif (i==7):
            test_algs(channel, "8_DepthFirstSearch", ["dfs_exercise.py", "dfs.py"])
        elif (i==8):
            test_algs(channel, "8_recursion", ["recursion.py"])
        elif (i==9):
            test_algs(channel, "9_BreadthFirstSearch", ["bfs.py", "bfs_exercise_solution.py"])
        #Random delay
        time.sleep(abs(random.gauss(5,4)))      #+- 0-10s
        i = random.uniform(0, 1)


    #Close the connection
    ssh.close()









#Execute a computational intensive program
def good_behaviour3(ssh):
    #Open Connection
    ssh.connect('192.168.1.1', username="labcom", password="labcom")
    channel = ssh.invoke_shell()

    time.sleep(abs(random.gauss(2,1)))                    #+- 0-3s
   
    channel.send("cd Project/NotImportantFiles/\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(1,1)))                    #+- 0-2s
    
    channel.send("ls -l\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(1,1)))                    #+- 0-2s
    
    channel.send("cd MD5/\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii")) 

    time.sleep(abs(random.gauss(1,1)))                    #+- 0-2s
    
    channel.send("ls -l\n")
    while not channel.recv_ready():
        time.sleep(0.5)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(3,2)))                    #+- 0-5s

    channel.send("python3 decriptmd5.py -f random.txt\n")
    while not channel.recv_ready():
        time.sleep(10)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(6,3)))                    #+- 3-10s

    channel.send("python3 decriptmd5.py -f random.txt\n")
    while not channel.recv_ready():
        time.sleep(10)
    out = channel.recv(9999)
    print(out.decode("ascii"))

    time.sleep(abs(random.gauss(6,3)))                    #+- 3-10s

    channel.send("python3 decriptmd5.py -f random.txt\n")
    while not channel.recv_ready():
        time.sleep(10)
    out = channel.recv(9999)
    print(out.decode("ascii"))
    
    
    #Close the connection
    ssh.close()







#Create SSH client
ssh = SSHClient()
ssh.load_system_host_keys()

i = random.uniform(0, 1)
while (i < 0.5):
    i = randrange(3)
    if i == 0:
        good_behaviour1(ssh)
    elif i == 1:
        good_behaviour2(ssh)
    elif i == 2:
        good_behaviour3(ssh)

    i = random.uniform(0, 1)
    #NOTE: FAZ SENTIDO TER AQUI UM? FALAR
    time.sleep(abs(random.gauss(10,5)))                    #+- 0-20



