


![alt text](https://i.ytimg.com/vi/pSqmAO-m7Lk/maxresdefault.jpg)


# CENG443 Heterogeneous Parallel Programing - Final project

- **Subject:** Paralleling Dijkstra's Shortesh Path Algorithm with Cuda 
- **Student:** Enes Furkan Fidan - 250201028
- **Instructor:** Işıl Öz

| Deadlines | Date   | Status    |
| :---:   | :---: | :---: |
| Github Page | 2 February | Done  |
| First Version | 15 February | In Review  |
| Meeting | 29 February | To-Do   |
| Final Version | 5 January | To-Do   |




# Problem Definition

In the implementation of shortest path algorithms, the same operations are repeated for more than one associated node and a continuous update is made. Continuously updating some data is not an efficient situation in parallel approaches.

In this project, I aim to observe that there will be some performance improvements by implementing GPU-based algorithm in a data-parallel way. When I examine some similar applications, I aim to observe that the performance efficiency of parallel implantation will be better, especially when the number of nodes exceeds a certain threshold.

## Dijkstra's Algorithm
With [Dijkstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm), you can find the shortest path between nodes in a graph. Particularly, you can find the shortest path from a node (called the "source node") to all other nodes in the graph, producing a shortest-path tree.

![alt text](https://upload.wikimedia.org/wikipedia/commons/5/57/Dijkstra_Animation.gif)

- Dijkstra's Algorithm basically starts at the node that you choose (the source node) and it analyzes the graph to find the shortest path between that node and all the other nodes in the graph.
- The algorithm keeps track of the currently known shortest distance from each node to the source node and it updates these values if it finds a shorter path.
- Once the algorithm has found the shortest path between the source node and another node, that node is marked as "visited" and added to the path.
- The process continues until all the nodes in the graph have been added to the path. This way, we have a path that connects the source node to all other nodes following the shortest path possible to reach each node.

**Serial Implementation Steps**

1- Initialize start node and set the distance to 0.

2- Initilize shortest path array, where the vertices with a shortest path found will be placed and place source node within.

3- From the nodes in the shortest path array, find the shortest neighbor node which is not visited.

4- Add this node to the shortest path array.

5- Update the minimum distance to each unvisited nodes.

6- Repeat from the step 3 until the destination reached, or until all nodes have been visited.

**Time Complexity of Serial Version is 𝑂(𝑛2).**


# Parallization

Parallel priority queue approach used in the final version. Multiple threads can access and update the priority queue to find the shortest paths.

# Problems
- **Data Dependencies:** Algorithm process relies on maintaning the correct distances between vertices and these distances can be updated as the algorithm proceeds.
- **Load Balancing:** It is important to ensure that the work is evenly distributed among threads. If it is not, some threads may idle while others overloaded.


