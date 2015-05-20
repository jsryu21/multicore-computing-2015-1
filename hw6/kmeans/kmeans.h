

#ifndef __KMENAS_H__
#define __KMEANS_H__

struct Point {
    Point() {}
    Point(float x_, float y_) : x(x_), y(y_) {
    }
    float x, y;
};


// Kmean algorighm
void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* clsfy_result, int num_threads, int local_size);

#endif // __KMEANS_H__

