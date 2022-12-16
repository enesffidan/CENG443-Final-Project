#include <iostream>
#include <queue>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const int INF = 1000000000;
const int NUM_VERTICES = 4;





// Structure to represent a node in the graph
struct Node {
  int distance;
  int predecessor;
};

// Structure to represent a vertex in the graph
struct Vertex {
  int x, y;
  Node data;
};

// Structure to represent an edge in the graph
struct Edge {
  int from, to;
  int weight;
};





// Function to allocate memory for the graph on the device
void allocateMemory(Vertex*& vertexArray, Edge*& edgeArray, int numVertices, int numEdges) {
  cudaMalloc((void**)&vertexArray, numVertices * sizeof(Vertex));
  cudaMalloc((void**)&edgeArray, numEdges * sizeof(Edge));
}

// Function to copy the data for the graph from the host to the device
void copyDataToDevice(Vertex* vertexArray, Edge* edgeArray, Vertex* d_vertexArray, Edge* d_edgeArray, int numVertices, int numEdges) {
  cudaMemcpy(d_vertexArray, vertexArray, numVertices * sizeof(Vertex), cudaMemcpyHostToDevice);
  cudaMemcpy(d_edgeArray, edgeArray, numEdges * sizeof(Edge), cudaMemcpyHostToDevice);

}


void device_info()
{
    int device_count;
    cudaGetDeviceCount(&device_count);

    // Iterate over the devices
    for (int i = 0; i < device_count; i++) {
        // Get the device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // Print the device name and model
        printf("Device %d: %s\n", i, prop.name);
        printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
	    printf("Maximum number of blocks: %d\n", prop.maxGridSize[3]);
    }
}


// Function to compute the shortest paths using Dijkstra's algorithm
__global__ void dijkstra(Vertex* d_vertexArray, Edge* d_edgeArray, int numVertices, int numEdges, int source) {
  // Create a priority queue to store the nodes to be processed
  int pq[NUM_VERTICES];
  int pqSize = 0;

  // Set the distance of the source vertex to 0 and enqueue it
  d_vertexArray[source].data.distance = 0;
  pq[pqSize++] = source;

  // Initialize the distances of all other vertices to infinity
  for (int i = 0; i < numVertices; i++) {
    if (i != source) {
      d_vertexArray[i].data.distance = INF;
    }
  }

  // Process the nodes in the priority queue
  while (pqSize > 0) {
    // Get the vertex with the smallest distance
    int u = pq[0];
    pqSize--;

    // Move the last element to the front of the queue
    pq[0] = pq[pqSize];

    // Sift the element down to its correct position in the queue
    int i = 0;
    while (i < pqSize) {
      int left = 2 * i + 1;
      int right = 2 * i + 2;
      int smallest = i;
      if (left < pqSize && d_vertexArray[pq[left]].data.distance < d_vertexArray[pq[i]].data.distance) {
            smallest = left;
      }
      if (right < pqSize && d_vertexArray[pq[right]].data.distance < d_vertexArray[pq[smallest]].data.distance) {
        smallest = right;
      }
      if (smallest != i) {
        int temp = pq[i];
        pq[i] = pq[smallest];
        pq[smallest] = temp;
        i = smallest;
      } else {
        break;
      }
    }

    // Process the edges of the vertex
    for (int i = 0; i < numEdges; i++) {
      int v = d_edgeArray[i].to;
      int weight = d_edgeArray[i].weight;

      // Relax the edge
      if (d_vertexArray[v].data.distance > d_vertexArray[u].data.distance + weight) {
        d_vertexArray[v].data.distance = d_vertexArray[u].data.distance + weight;
        d_vertexArray[v].data.predecessor = u;

        // Update the position of the vertex in the priority queue
        for (int j = 0; j < pqSize; j++) {
          if (pq[j] == v) {
            i = j;
            break;
          }
        }
        while (i > 0 && d_vertexArray[pq[(i - 1) / 2]].data.distance > d_vertexArray[pq[i]].data.distance) {
          int temp = pq[(i - 1) / 2];
          pq[(i - 1) / 2] = pq[i];
          pq[i] = temp;
          i = (i - 1) / 2;
        }
      }
    }
  }
}


int main() {
  // Initialize the graph data
  //Vertex vertexArray[] = {{0, 0, {0, -1}}, {1, 0, {INF, -1}}, {2, 0, {INF, -1}}, {3, 0, {INF, -1}}};
  Vertex vertexArray[] = {{0, 0, {0, -1}}, {1, 0, {3, 1}}, {2, 0, {7, 2}}, {3, 0, {11, 1}}};
  Edge edgeArray[] = {{0, 1, 10}, {0, 2, 5}, {1, 2, 2}, {1, 3, 1}, {2, 3, 3}};
  int numVertices = sizeof(vertexArray) / sizeof(Vertex);
  int numEdges = sizeof(edgeArray) / sizeof(Edge);




  // Allocate memory for the graph on the device
  Vertex* d_vertexArray;
  Edge* d_edgeArray;
  allocateMemory(d_vertexArray, d_edgeArray, numVertices, numEdges);

  // Copy the data for the graph from the host to the device
  copyDataToDevice(vertexArray, edgeArray, d_vertexArray, d_edgeArray, numVertices, numEdges);

  // Compute the shortest paths using Dijkstra's algorithm
  dijkstra<<<1, 1>>>(d_vertexArray, d_edgeArray, numVertices, numEdges, 0);

  // Copy the data for the graph from the device back to the host
  cudaMemcpy(vertexArray, d_vertexArray, numVertices * sizeof(Vertex), cudaMemcpyDeviceToHost);

  //Print device info
  device_info();

  // Print the shortest path from the source to each vertex
  for (int i = 0; i < numVertices; i++) {
    printf("Vertex %d: distance = %d, predecessor = %d\n", i, vertexArray[i].data.distance, vertexArray[i].data.predecessor);
  }

  // Free the memory for the graph on the device
  cudaFree(d_vertexArray);
  cudaFree(d_edgeArray);

  return 0;
}