

#ifndef __KMENAS_H__
#define __KMEANS_H__

struct Point {
    float x, y;
};


// Kmean algorighm
void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* clsfy_result, int num_threads);

#endif // __KMEANS_H__

