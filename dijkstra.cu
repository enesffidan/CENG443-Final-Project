#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

#define INF INT_MAX
#define NUM_VERTICES 1023
#define NUM_EDGES 1023


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
void allocateMemory(Vertex** vertexArray, Edge** edgeArray, int numVertices, int numEdges) {
  cudaMalloc((void**)vertexArray, numVertices * sizeof(Vertex));
  cudaMalloc((void**)edgeArray, numEdges * sizeof(Edge));
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
    // Get the thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // Create a shared memory priority queue to store the nodes to be processed
    __shared__ int pq[NUM_VERTICES];
    __shared__ int pqSize;
    // Set the distance of the source vertex to 0 and enqueue it
    if (tid == 0) {
        d_vertexArray[source].data.distance = 0;
        pq[0] = source;
        pqSize = 1;
    }
    // Initialize the distances of all other vertices to infinity
    __syncthreads();
    if (tid < numVertices) {
        if (tid != source) {
            d_vertexArray[tid].data.distance = INF;
        }
    }
    // Process the nodes in the priority queue
    while (pqSize > 0) {
        // Get the vertex with the smallest distance
        int u;
        if (tid == 0) {
            u = pq[0];
            pqSize--;
        }
        __syncthreads();
        // Move the last element to the front of the queue
        if (tid == 0) {
            pq[0] = pq[pqSize];
        }
        // Sift the element down to its correct position in the queue
        __syncthreads();
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

        // Update the distances of the neighbors of u
        for (int j = 0; j < numEdges; j++) {
            if (d_edgeArray[j].from == u) {
                int v = d_edgeArray[j].to;
                int weight = d_edgeArray[j].weight;
                if (d_vertexArray[u].data.distance + weight < d_vertexArray[v].data.distance) {
                    d_vertexArray[v].data.distance = d_vertexArray[u].data.distance + weight;
                    d_vertexArray[v].data.predecessor = u;
                    // Check if v should be enqueued
                    bool shouldEnqueue = false;
                    if (tid == 0) {
                        shouldEnqueue = true;
                        for (int k = 0; k < pqSize; k++) {
                            if (pq[k] == v) {
                                shouldEnqueue = false;
                                break;
                            }
                        }
                    }
                    __syncthreads();
                    if (shouldEnqueue) {
                        pq[pqSize++] = v;
                    }
                    pq[pqSize++] = v;
                }
            }
        }
    }
}


int main() {
  // Create the vertex and edge arrays on the host
    Vertex vertexArray[NUM_VERTICES];
    Edge edgeArray[NUM_EDGES];
  
  for (int i = 0; i < NUM_VERTICES; i++) {
    vertexArray[i].x = i;
    vertexArray[i].y = i+1;
    vertexArray[i].data.distance = 0;
    vertexArray[i].data.predecessor = -1;
}
for (int i = 0; i < NUM_EDGES; i++) {
    edgeArray[i].from = i;
    edgeArray[i].to = i+1;
    edgeArray[i].weight = i;
}
  int numVertices = sizeof(vertexArray) / sizeof(Vertex);
  int numEdges = sizeof(edgeArray) / sizeof(Edge);

  //Timer initialization
  float runningTime;
  cudaEvent_t timeStart, timeEnd;
 //Creates the timers
  cudaEventCreate(&timeStart);
  cudaEventCreate(&timeEnd);
  
  //Start the timer
  cudaEventRecord(timeStart, 0);


  // Allocate memory for the graph on the device
  Vertex* d_vertexArray;
  Edge* d_edgeArray;
  allocateMemory(&d_vertexArray, &d_edgeArray, numVertices, numEdges);

  // Copy the data for the graph from the host to the device
  copyDataToDevice(vertexArray, edgeArray, d_vertexArray, d_edgeArray, numVertices, numEdges);

  // Compute the shortest paths using Dijkstra's algorithm
  //dijkstra<<<1, numVertices>>>(d_vertexArray, d_edgeArray, numVertices, numEdges, 0);
  dijkstra<<<1, 1024>>>(d_vertexArray, d_edgeArray, numVertices, numEdges, 0);

  //Timing Events
  cudaEventRecord(timeEnd, 0);
  cudaEventSynchronize(timeEnd);
  cudaEventElapsedTime(&runningTime, timeStart, timeEnd);


  // Copy the data for the graph from the device to the host
  cudaMemcpy(vertexArray, d_vertexArray, numVertices * sizeof(Vertex), cudaMemcpyDeviceToHost);

  printf("Current Number of Blocks: %d Current Number of Threads:%d\n",1,1);

  // Print the shortest paths
  for (int i = 0; i < numVertices; i++) {
    printf("Vertex %d: Distance = %d, Predecessor = %d\n", i, vertexArray[i].data.distance, vertexArray[i].data.predecessor);
  }

    //Running Time
  printf("Running Time: %f ms\n", runningTime);
  device_info();

  // Free the memory allocated on the device
  cudaFree(d_vertexArray);
  cudaFree(d_edgeArray);
  cudaEventDestroy(timeStart);
  cudaEventDestroy(timeEnd);

  return 0;
}
