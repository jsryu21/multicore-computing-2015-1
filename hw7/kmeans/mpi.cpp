#include "kmeans.h"
#include "mpi.h"
#include <cfloat>

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned, int num_threads, int local_size, int argc, char** argv)
{
    int numnodes, myid;
    int mpi_err = MPI_Init(&argc, &argv);
    mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &numnodes);
    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    // Count number of data in each class
    int* count = new int[class_n];
    Point* tempCentroids = new Point[class_n];
    int* tempCount = new int[class_n];
    // Iterate through number of interations
    for (int i = 0; i < iteration_n; i++) {
        memset(tempCentroids, 0, class_n * sizeof(Point));
        memset(tempCount, 0, class_n * sizeof(int));
        // Assignment step
        for (int data_i = 0; data_i < data_n; ++data_i) {
            float min_dist = FLT_MAX;
            for (int class_i = 0; class_i < class_n; class_i++) {
                float x = data[data_i].x - centroids[class_i].x;
                float y = data[data_i].y - centroids[class_i].y;
                float dist = x * x + y * y;
                if (dist < min_dist) {
                    partitioned[data_i] = class_i;
                    min_dist = dist;
                }
            }
            // Sum up and count data for each class
            int index = class_n + partitioned[data_i];
            tempCentroids[index].x += data[data_i].x;
            tempCentroids[index].y += data[data_i].y;
            tempCount[index]++;
        }
        // Update step
        {
            // Clear sum buffer and class count
            memset(centroids, 0, class_n * sizeof(Point));
            memset(count, 0, class_n * sizeof(int));
        }
        for (int class_i = 0; class_i < class_n; ++class_i) {
            for (int t = 0; t < num_threads; ++t) {
                centroids[class_i].x += tempCentroids[t * class_n + class_i].x;
                centroids[class_i].y += tempCentroids[t * class_n + class_i].y;
                count[class_i] += tempCount[t * class_n + class_i];
            }
            // Divide the sum with number of class for mean point
            centroids[class_i].x /= count[class_i];
            centroids[class_i].y /= count[class_i];
        }
    }
    delete[] tempCount;
    delete[] tempCentroids;
    delete[] count;
}
