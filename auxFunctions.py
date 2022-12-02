def test_algs1(shell):
    shell.run(["cd", "1_BinarySearch"])
    shell.run(["python3", "binary_search_exercise_solution.py"])
    shell.run(["python3", "binarysearch.py"])
    shell.run(["cd", ".."])

def test_algs2(shell):
    shell.run(["cd", "2_BubbleSort"])
    shell.run(["python3", "bubble_sort_exercise_solution.py"])
    shell.run(["python3", "bubble_sort.py"])
    shell.run(["cd", ".."])

def test_algs3(shell):
    shell.run(["cd", "3_QuickSort"])
    shell.run(["python3", "quick_sort_exercise_soluiton_lomuto.py"])
    shell.run(["python3", "quick_sort.py"])
    shell.run(["cd", ".."])

def test_algs4(shell):
    shell.run(["cd", "4_InsertionSort"])
    shell.run(["python3", "insertion_sort_exercise_solution.py"])
    shell.run(["python3", "insertion_sort.py"])
    shell.run(["cd", ".."])

def test_algs5(shell):
    shell.run(["cd", "5_MergeSort"])
    shell.run(["python3", "merge_sort_exercise_solution.py"])
    shell.run(["python3", "merge_sort_primitive.py"])
    shell.run(["python3", "merge_sort_final.py"])
    shell.run(["cd", ".."])

def test_algs6(shell):
    shell.run(["cd", "6_ShellSort"])
    shell.run(["python3", "shell_sort.py"])
    shell.run(["python3", "shell_sort_exercise_solution.py"])
    shell.run(["cd", ".."])

def test_algs7(shell):
    shell.run(["cd", "7_SelectionSort"])
    shell.run(["python3", "selection_sort_exercise_solution.py"])
    shell.run(["python3", "selection_sort.py"])
    shell.run(["cd", ".."])

def test_algs8(shell):
    shell.run(["cd", "8_DepthFirstSearch"])
    shell.run(["python3", "dfs_exercise.py"])
    shell.run(["python3", "dfs.py"])
    shell.run(["cd", ".."])

def test_algs9(shell):
    shell.run(["cd", "8_recursion"])
    shell.run(["python3", "recursion.py"])
    shell.run(["cd", ".."])

def test_algs10(shell):
    shell.run(["cd", "9_BreadthFirstSearch"])
    shell.run(["python3", "bfs.py"])
    shell.run(["python3", "bfs_exercise_solution.py"])
    shell.run(["cd", ".."])