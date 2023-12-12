# BSC IT AI Practical

Practical 1 

Write a program to implement depth first search algorithm.
```py
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    print(start, end=' ')

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F', 'G'],
    'D': ['B'],
    'E': ['B'],
    'F': ['C'],
    'G': ['C']
}

# Start DFS from node 'A'
print("DFS traversal starting from node 'A':")
dfs(graph, 'A')

```

Breath First Search (BFS)
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

Write a program for Hill climbing problem.
```py
list = [2,4,5,6,7,8,9,10,3,4,5,6,7,8,9,34,36,23,21,45]
output = []
previous = None
for x in range(len(list)):
    if x > 0:
        previous = list[x-1]
        if list[x-1] >= list[x]:
            break
    output.append(list[x])

print(output)
```
Output:
```
[2, 4, 5, 6, 7, 8, 9, 10]
```

Write a program to implement A* algorithm.
```py
import heapq

def astar(grid, start, goal):
    def heuristic(node):
        x1, y1 = node
        x2, y2 = goal
        return abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance

    open_set = [(0, start)]  # Priority queue with initial cost and node
    came_from = {}
    g_score = {node: float('inf') for node in grid}
    g_score[start] = 0

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor in neighbors(current):
            tentative_g_score = g_score[current] + 1  # Assuming each step has a cost of 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor)

                if neighbor not in [node for _, node in open_set]:
                    heapq.heappush(open_set, (f_score, neighbor))

    return None  # No path found

def neighbors(node):
    x, y = node
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

# Example usage
if __name__ == "__main__":
    grid = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)}
    start = (0, 0)
    # start = (0,2)
    goal = (3, 3)

    path = astar(grid, start, goal)

    if path:
        print("Shortest Path:", path)
    else:
        print("No path found.")
```
A* Output
```
Shortest Path: [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)]
```

Write a program to implement AO* algorithm.
```py
import heapq

def astar(grid, start, goal):
    def heuristic(node):
        x1, y1 = node
        x2, y2 = goal
        return abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance

    open_set = [(0, start)]  # Priority queue with initial cost and node
    came_from = {}
    g_score = {node: float('inf') for node in grid}
    g_score[start] = 0

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor in neighbors(current):
            tentative_g_score = g_score[current] + 1  # Assuming each step has a cost of 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor)

                if neighbor not in [node for _, node in open_set]:
                    heapq.heappush(open_set, (f_score, neighbor))

    return None  # No path found

def neighbors(node):
    x, y = node
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

# Example usage
if __name__ == "__main__":
    grid = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)}
    # start = (0, 0)
    start = (0,2)
    goal = (3, 3)

    path = astar(grid, start, goal)

    if path:
        print("Shortest Path:", path)
    else:
        print("No path found.")
```

AO* output
```
Shortest Path: [(0, 2), (0, 3), (1, 3), (2, 3), (3, 3)]
```

Design an application to simulate number puzzle problem (8 puzzle game)
```py
from collections import deque

def get_neighbors(state):
    moves = [1, -1, 3, -3]
    empty_index = state.index(0)
    neighbors = []
    for move in moves:
        new_index = empty_index + move
        if 0 <= new_index < 9:
            neighbor = state[:]
            neighbor[empty_index], neighbor[new_index] = neighbor[new_index], neighbor[empty_index]
            neighbors.append(neighbor)
    return neighbors

def bfs(initial_state, goal_state):
    visited, queue = set(), deque([(initial_state, [])])
    while queue:
        state, path = queue.popleft()
        if state == goal_state: return path
        for neighbor in get_neighbors(state):
            if tuple(neighbor) not in visited: visited.add(tuple(neighbor)); queue.append((neighbor, path + [neighbor]))
    return None

initial_state = [1, 2, 3, 0, 4, 6, 7, 5, 8]
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]

solution_path = bfs(initial_state, goal_state)
if solution_path:
    print("Solution path:", solution_path)
else:
    print("No solution found.")
```
```
Solution path: [[1, 2, 3, 4, 0, 6, 7, 5, 8], [1, 2, 3, 4, 5, 6, 7, 0, 8], [1, 2, 3, 4, 5, 6, 7, 8, 0]]
```

Write a program to solve water jug problem.
```py
from collections import deque
def water_jug_bfs(a, b, target):
    visited = set()
    q = deque([(0, 0)])
    parent = {}
    while q:
        u = q.popleft()
        if u in visited:
            continue
        visited.add(u)
        if u[0] == target or u[1] == target:
            path = []
            while u != (0, 0):
                path.append(u)
                u = parent[u]
            path.append((0, 0))
            path.reverse()
            for state in path:
                print(state)
            return
        next_states = [
            (u[0], b),  # Fill Jug2
            (a, u[1]),  # Fill Jug1
            (u[0] + min(u[1], a - u[0]), u[1] - min(u[1], a - u[0])),  # Pour Jug2 to Jug1
            (u[0] - min(u[0], b - u[1]), u[1] + min(u[0], b - u[1])),  # Pour Jug1 to Jug2
            (a, 0),  # Empty Jug2
            (0, b)   # Empty Jug1
        ]
        for v in next_states:
            if 0 <= v[0] <= a and 0 <= v[1] <= b and v not in visited:
                q.append(v)
                parent[v] = u
    print("No solution")

if __name__ == '__main__':
    Jug1, Jug2, target = 4, 3, 2
    print("Path from initial state to solution state ::")
    water_jug_bfs(Jug1, Jug2, target)
```
output
```
Path from initial state to solution state ::
(0, 0)
(0, 3)
(3, 0)
(3, 3)
(4, 2)
```
Write a program to solve Missionaries and Cannibals problem.
```py
lM, lC, rM, rC, userM, userC, k = 3,3,0,0,0,0,0
print("\nM M M C C C |	 --- | \n")
def printState(arrow):
    print("\n")
    for i in range(0, lM):
        print("M ", end="")
    for i in range(0, lC):
        print("C ", end="")
    print(arrow, end="")
    for i in range(0, rM):
        print("M ", end="")
    for i in range(0, rC):
        print("C ", end="")
    print("\n")
try:
    while (True):
        while (True):
            print("Left side -> right side river travel")
            uM = int(input("Enter number of Missionaries travel => "))
            uC = int(input("Enter number of Cannibals travel => "))
            if ((uM == 0) and (uC == 0)):
                print("Empty travel not possible")
                print("Re-enter : ")
            elif (((uM + uC) <= 2) and ((lM - uM) >= 0) and ((lC - uC) >= 0)):
                break
            else:
                print("Wrong input re-enter : ")
        lM = (lM - uM)
        lC = (lC - uC)
        rM += uM
        rC += uC
        printState("| --> | ")
        k += 1
        if (((lC == 3) and (lM == 1)) or ((lC == 3) and (lM == 2)) or ((lC == 2) and (lM == 1)) or (
                (rC == 3) and (rM == 1)) or ((rC == 3) and (rM == 2)) or ((rC == 2) and (rM == 1))):
            print("Cannibals eat missionaries:\nYou lost the game")
            break
        if ((rM + rC) == 6):
            print("You won the game : \n\tCongrats")
            print("Total attempt")
            print(k)
            break
        while (True):
            print("Right side -> Left side river travel")
            userM = int(input("Enter number of Missionaries travel => "))
            userC = int(input("Enter number of Cannibals travel => "))
            if ((userM == 0) and (userC == 0)):
                print("Empty travel not possible")
                print("Re-enter : ")
            elif (((userM + userC) <= 2) and ((rM - userM) >= 0) and ((rC - userC) >= 0)):
                break
            else:
                print("Wrong input re-enter : ")
        lM += userM
        lC += userC
        rM -= userM
        rC -= userC
        k += 1
        printState("| <-- |")
        if (((lC == 3) and (lM == 1)) or ((lC == 3) and (lM == 2)) or ((lC == 2) and (lM == 1)) or (
                (rC == 3) and (rM == 1)) or ((rC == 3) and (rM == 2)) or ((rC == 2) and (rM == 1))):
            print("Cannibals eat missionaries:\nYou lost the game")
            break
except EOFError as e:
    print("\nInvalid input please retry !!")
```

Write a program to shuffle Deck of cards.
```py
import  itertools, random
deck = list(itertools.product(range(1,14), ["spade", "heart", "diamond", "club"]))
random.shuffle(deck)
print("you got")
for x in range(5):
    print(deck[x][0], "of", deck[x][1])
```

Solve traveling salesman problem using artificial intelligence technique.
```py
from sys import maxsize
from itertools import permutations

V = 4


# implementation of traveling Salesman Problem
def travellingSalesmanProblem(graph, s):
    # store all vertex apart from source vertex
    vertex = []
    for i in range(V):
        if i != s:
            vertex.append(i)

    # store minimum weight Hamiltonian Cycle
    min_path = maxsize
    next_permutation = permutations(vertex)
    for i in next_permutation:

        # store current Path weight(cost)
        current_pathweight = 0

        # compute current path weight
        k = s
        for j in i:
            current_pathweight += graph[k][j]
            k = j
        current_pathweight += graph[k][s]

        # update minimum
        min_path = min(min_path, current_pathweight)

    return min_path


# Driver Code
if __name__ == "__main__":
    # matrix representation of graph
    graph = [[0, 10, 15, 20], [10, 0, 35, 25],
             [15, 35, 0, 30], [20, 25, 30, 0]]
    s = 0
    print(travellingSalesmanProblem(graph, s))
```

word problem | minimum jum to reach given distance
```py
class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# To represent line of (ax + by + c)format


class line:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

# Returns 1 if evaluation is greater > 0,
# else returns -1


def evalPointOnLine(p, curLine):
    eval = curLine.a * p.x + curLine.b * p.y + curLine.c
    if (eval > 0):
        return 1
    return -1

# Returns minimum jumps to reach
# dest point from start point


def minJumpToReachDestination(start, dest, lines, N):
    jumps = 0
    for i in range(N):
        # get sign of evaluation from point
        # co-ordinate and line equation
        signStart = evalPointOnLine(start, lines[i])
        signDest = evalPointOnLine(dest, lines[i])

        # if both evaluation are of opposite sign,
        # increase jump by 1
        if (signStart * signDest < 0):
            jumps = jumps + 1

    return jumps


# Driver code to test above methods
start = point(1, 1)
dest = point(-2, -1)

lines = []
lines.append(line(1, 0, 0))
lines.append(line(0, 1, 0))
lines.append(line(1, 1, -2))

print(minJumpToReachDestination(start, dest, lines, 3))
```

Solve constraint satisfaction problem
```cmd
pip install python-constraint
```
```py
from constraint import *
def constant_function(a, b, c):
    return a + b == c
problem = Problem()
problem.addVariable('A', range(1, 11))
problem.addVariable('B', range(1, 11))
problem.addVariable('C', range(1, 11))
problem.addConstraint(constant_function, ('A', 'B', 'C'))
solutions = problem.getSolutions()
for solution in solutions:
    print(f'A={solution["A"]}, B={solution["B"]}, C={solution["C"]}')
```
Derive the expressions based on Associative law
```py
def associative_law_add(a,b,c):
    la = (a+b)+c
    ra = a+(b+c)
    if la == ra:
        print(f'The addition is associative: ({a}+{b})+{c} = {a} + ({b} + {c})')
    else:
        print(f'The addition is not associative: ({a}+{b})+{c} = {a} + ({b} + {c})')
associative_law_add(3,5,7)
```
Derive the expressions based on Distributive law
```cmd
pip install sympy
```
```py
import sympy as sp
a,b,c = sp.symbols('a b c')
original_expr =  a*(b+c)
distributive_expr = sp.expand(original_expr)
print("Original Expression:", original_expr)
print("Expression after applying distributive law:", distributive_expr)
```

Write a program to derive the predicate.
```py
prd = input("Enter a predicate:")
prd2 = input("Enter a predicate:")
arg = input("Enter an argument:")
print(prd + "->" + arg)
print(arg + "->" + prd2)
print(prd + "->" + prd2)
```
