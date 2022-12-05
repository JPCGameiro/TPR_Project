import spur
import time

#Auxiliary function to run algogithms
def test_algs(shell, dir, program_list):
	shell.run(["cd", dir])
	for p in program_list:
		shell.run(["python3", p])
	shell.run(["cd", ".."])



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
    
    test_algs(shell, "1_BinarySearch", ["binary_search_exercise_solution.py", "binarysearch.py"])
    test_algs(shell, "2_BubbleSort", ["bubble_sort_exercise_solution.py", "bubble_sort.py"])
    test_algs(shell, "3_QuickSort", ["quick_sort_exercise_soluiton_lomuto.py", "quick_sort.py"])
    test_algs(shell, "4_InsertionSort", ["insertion_sort_exercise_solution.py", "insertion_sort.py"])
    test_algs(shell, "5_MergeSort", ["merge_sort_exercise_solution.py", "merge_sort_primitive.py", "merge_sort_final.py"])
    test_algs(shell, "6_ShellSort", ["shell_sort.py", "shell_sort_exercise_solution.py"])
    test_algs(shell, "7_SelectionSort", ["selection_sort_exercise_solution.py", "selection_sort.py"])
    test_algs(shell, "8_DepthFirstSearch", ["dfs_exercise.py", "dfs.py"])
    test_algs(shell, "8_recursion", ["recursion.py"])
    test_algs(shell, "9_BreadthFirstSearch", ["bfs.py", "bfs_exercise_solution.py"])

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

