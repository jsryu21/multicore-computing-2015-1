#include "kmeans.h"
#include <cfloat>
#include <vector>
#include <omp.h>
#include <iostream>
#include <cstring>

int kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned, int num_threads, int local_size, int argc, char** argv)
{
    // Count number of data in each class
    int* count = new int[class_n];
    int max_threads = omp_get_max_threads();
    Point* tempCentroids = new Point[max_threads * class_n];
    int* tempCount = new int[max_threads * class_n];
    // Iterate through number of interations
    omp_set_num_threads(num_threads);
    for (int i = 0; i < iteration_n; i++) {
#pragma omp parallel
        {
            const int ithread = omp_get_thread_num();
#pragma omp single
            {
                memset(tempCentroids, 0, max_threads * class_n * sizeof(Point));
                memset(tempCount, 0, max_threads * class_n * sizeof(int));
            }
            // Assignment step
#pragma omp for
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
                int index = ithread * class_n + partitioned[data_i];
                tempCentroids[index].x += data[data_i].x;
                tempCentroids[index].y += data[data_i].y;
                tempCount[index]++;
            }
            // Update step
#pragma omp single
            {
                // Clear sum buffer and class count
                memset(centroids, 0, class_n * sizeof(Point));
                memset(count, 0, class_n * sizeof(int));
            }
#pragma omp for
            for (int class_i = 0; class_i < class_n; ++class_i) {
                for (int t = 0; t < max_threads; ++t) {
                    centroids[class_i].x += tempCentroids[t * class_n + class_i].x;
                    centroids[class_i].y += tempCentroids[t * class_n + class_i].y;
                    count[class_i] += tempCount[t * class_n + class_i];
                }
                // Divide the sum with number of class for mean point
                centroids[class_i].x /= count[class_i];
                centroids[class_i].y /= count[class_i];
            }
        }
    }
    delete[] tempCount;
    delete[] tempCentroids;
    delete[] count;
    return 0;
}
