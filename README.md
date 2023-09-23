# BSC IT AI Practical

Practical 1 - Breath First Search (BFS)
![image](https://github.com/Omkar270048/BSC_IT_AI_Practical/assets/69665958/560d211b-a56a-46ce-b762-ffdacae20ced)
```py
graph = {
    'a':['b','c','d'],
    'b':['e'],
    'c':['d','e'],
    'd':[],
    'e':[]
}

visited = set() #defining visitednas a set of elements, initially it is empty
def dfs(visited, graph, rootNode):
    if rootNode not in visited: #if rootNode is not in visited set, then add it.
        print(rootNode, end=" ")
        visited.add(rootNode)
        for neighbour in graph[rootNode]: #for element 'a', graph[rootNode] =  ['b', 'c', 'd']. 
            dfs(visited, graph, neighbour)  #now neighbour = 'b', it will run dfs function for 'b'. i.e dfs(visited, graph, 'b')
            
dfs(visited, graph, 'a')
```
BFS Algorithm (Breath first search algorithm)
![image](https://github.com/Omkar270048/BSC_IT_AI_Practical/assets/69665958/6ad726e2-bcbd-43b8-95ab-68cee77a4dd5)
```py
graph = [
    ['a', ['b', 'c']],
    ['b', ['d','e']],
    ['c', ['f', 'g']]
]
arr = []

for i in graph:
    for j in i:
        for k in j:
            if k not in arr:
                arr.append(k)
                print(k, end=" ")
```
4 Queen Problem (n Queen)
![image](https://github.com/Omkar270048/BSC_IT_AI_Practical/assets/69665958/491f5f60-3d14-4c1c-acbc-e81c30a76e84)
```py
def is_safe(board, row, col):
    for i in range(col):
        if board[row][i] == 1:
            return False
    
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    
    for i, j in zip(range(row, len(board), -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    
    return True

def solve_n_queens_util(board, col):
    if col >= len(board):
        return True
    
    for i in range(len(board)):
        if is_safe(board, i, col):
            board[i][col] = 1
            if solve_n_queens_util(board, col + 1):
                return True
            
            #backtrack
            board[i][col] = 0
    
    return False

def solve_n_queens():
    board = [[0 for _ in range(4)] for _ in range(4)]
    if not solve_n_queens_util(board, 0):
        print("No solution exists")
        return
    
    for row in board:
        print(row)

solve_n_queens()
```
Write a program to solve tower of Hanoi problem.
![image](https://github.com/Omkar270048/BSC_IT_AI_Practical/assets/69665958/0e0a6fde-8f11-4d5b-bebc-89340f96d1c9)
```py
def tower_of_hanoi(height, tower_A, tower_B, tower_C):
    if height == 1:
        print(f"Move disk 1 from {tower_A} to {tower_C}")
        return
    tower_of_hanoi(height-1, tower_A, tower_C, tower_B)
    print(f"Move disk {height} from {tower_A} to {tower_C}")
    tower_of_hanoi(height-1, tower_B, tower_A, tower_C)

height = int(input("Enter the number of disks: "))
tower_of_hanoi(height, 'tower_A', 'tower_B', 'tower_C')
```
Alpha Beta Puring Algorithm</br>
step 1:
![image](https://github.com/Omkar270048/BSC_IT_AI_Practical/assets/69665958/1b86b1d5-c0a0-4f3b-b602-8b032cb78dcb)
step 2:
![image](https://github.com/Omkar270048/BSC_IT_AI_Practical/assets/69665958/34a4ed3e-0ac7-48ad-a7b4-b7b7711131eb)
step 3:
![image](https://github.com/Omkar270048/BSC_IT_AI_Practical/assets/69665958/2c582ac9-a82a-4e1b-8252-7c0f0468b6bb)
```py
list = [3,5,6,3,1,2,0,-1]
flag = 'max'

def alphaBeta(start, size):
    global list, flag
    midIndex = size//2

    if size > 1:
        if flag=="max":
            flag="min"
            num = max(list[start], list[midIndex])
            start = list.index(num)
            alphaBeta(start, midIndex)
        else:
            flag = "max"
            num = min(list[start], list[midIndex])
            start = list.index(num)
            alphaBeta(start, midIndex)
    else:
        print(list[start])

alphaBeta(0, len(list))
```
