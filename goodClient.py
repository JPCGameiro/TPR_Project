import spur
import time


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
    shell.run(["echo",">", "", ""])
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




def good_behaviour2(shell):
    #Create the connection
    shell = spur.SshShell(hostname="192.168.1.1", username="labcom", password="labcom")

    shell.run(["cd", "Project"])
    shell.run(["cd", "NotImportantFiles"])
    shell.run(["cd", "data-structures-algorithms-python-master"])
    shell.run(["cd", "algorithms"])
    shell.run(["cd", "1_BinarySearch"])
    shell.run(["python3", "binary_search_exercise_solution.py"])
    shell.run(["python3", "binarysearch.py"])
    shell.run(["cd", ".."])
    shell.run(["cd", "2_BubbleSort"])
    shell.run(["python3", "bubble_sort_exercise_solution.py"])
    shell.run(["python3", "bubble_sort.py"])
    shell.run(["cd", ".."])
    shell.run(["cd", "3_QuickSort"])
    shell.run(["python3", "quick_sort_exercise_soluiton_lomuto.py"])
    shell.run(["python3", "quick_sort.py"])
    shell.run(["cd", ".."])
    shell.run(["cd", "4_InsertionSort"])
    shell.run(["python3", "insertion_sort_exercise_solution.py"])
    shell.run(["python3", "insertion_sort.py"])
    shell.run(["cd", ".."])
    shell.run(["cd", "5_MergeSort"])
    shell.run(["python3", "merge_sort_exercise_solution.py"])
    shell.run(["python3", "merge_sort_primitive.py"])
    shell.run(["python3", "merge_sort_final.py"])
    shell.run(["cd", ".."])
    shell.run(["cd", "6_ShellSort"])
    shell.run(["python3", "shell_sort.py"])
    shell.run(["python3", "shell_sort_exercise_solution.py"])
    shell.run(["cd", ".."])
    shell.run(["cd", "7_SelectionSort"])
    shell.run(["python3", "selection_sort_exercise_solution.py"])
    shell.run(["python3", "selection_sort.py"])
    shell.run(["cd", ".."])
    shell.run(["cd", "8_DepthFirstSearch"])
    shell.run(["python3", "dfs_exercise.py"])
    shell.run(["python3", "dfs.py"])
    shell.run(["cd", ".."])
    shell.run(["cd", "8_recursion"])
    shell.run(["python3", "recursion.py"])
    shell.run(["cd", ".."])
    shell.run(["cd", "9_BreadthFirstSearch"])
    shell.run(["python3", "bfs.py"])
    shell.run(["python3", "bfs_exercise_solution.py"])
    shell.run(["cd", "../../../../.."])

    #Close the connection
    shell.close()



def good_behavior3(shell):
    pass

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

