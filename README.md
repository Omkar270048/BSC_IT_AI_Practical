# BSC_IT_AI_Practical

Practical 1
```py
import collections
import queue
def bfs(graph, root):
    visited = set()
    queue = collections.deque([root])
    while queue:
        vertex = queue.popleft()
        visited.add(vertex)
        for i in graph[vertex]:
            if i not in visited:
                queue.append(i)
    print(visited)
__name__=="__main__"
graph={0:[1,2,3], 1:[0, 2], 2:[0, 1, 4], 3:[0], 4:[2]}
bfs(graph,0)
```
