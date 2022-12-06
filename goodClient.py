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
    
    stdin, stdout, stderr = ssh.exec_command('ls -l; cd Project/; ls -l; cd NotImportantFiles/; ls -l')
    print(stdout.read())
    #Random Delay

    stdin, stdout, stderr = ssh.exec_command('ls -l; cd Project/; ls -l; cd NotImportantFiles/; ls -l; rm -rf helloworld.py')
    print(stdout.read())
    #Random Delay

    stdin, stdout, stderr = ssh.exec_command('cd Project/NotImportantFiles/; touch helloworld.py')
    print(stdout.read())
    #Random Delay
 
    stdin, stdout, stderr = ssh.exec_command('cd Project/NotImportantFiles/; ls; echo "print(\'helloFworld\')" >> helloworld.py')
    print(stdout.read())
    #Random Delay

    stdin, stdout, stderr = ssh.exec_command('cd Project/NotImportantFiles/; python3 helloworld.py')
    print(stdout.read())
    #Random Delay

    #Close the connection
    ssh.close()





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

    #shell.run(["cd", "../../../.."])

    #Close the connection
    ssh.close()

def good_behavior3(shell):
    pass




#Create SSH client
ssh = SSHClient()
ssh.load_system_host_keys()

#good_behaviour1(ssh)
good_behaviour2(ssh)
