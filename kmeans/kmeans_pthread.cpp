
#include "kmeans.h"
#include <cstdlib>
#include <cfloat>
#include <pthread.h>
#include <vector>
#include <cmath>
#include <iostream>

struct AssignThreadData {
    AssignThreadData(int class_n, const Point* centroids, const Point* data, int* partitioned);
    Point* t;
    int class_n;
    const Point* centroids;
    const Point* data;
    int* partitioned;
    int data_start_index;
    int data_end_index;
};

AssignThreadData::AssignThreadData(int class_n, const Point* centroids, const Point* data, int* partitioned) {
    this->class_n = class_n;
    this->centroids = centroids;
    this->data = data;
    this->partitioned = partitioned;
}

struct UpdateThreadData {
};

void* AssignmentWork(void* threadarg) {
    AssignThreadData* my_data = static_cast< AssignThreadData* >(threadarg);
    Point& t = *(my_data->t);
    int class_n = my_data->class_n;
    const Point* centroids = my_data->centroids;
    const Point* data = my_data->data;
    int* partitioned = my_data->partitioned;
    int data_start_index = my_data->data_start_index;
    int data_end_index = my_data->data_end_index;

    for (int data_i = data_start_index; data_i < data_end_index; ++data_i) {
        float min_dist = DBL_MAX;
        for (int class_i = 0; class_i < class_n; class_i++) {
            t.x = data[data_i].x - centroids[class_i].x;
            t.y = data[data_i].y - centroids[class_i].y;
            float dist = t.x * t.x + t.y * t.y;
            if (dist < min_dist) {
                partitioned[data_i] = class_i;
                min_dist = dist;
            }
        }
    }

    pthread_exit(static_cast< void* >(my_data));
}

void* UpdateWork(void* threadarg) {
    UpdateThreadData* my_data = static_cast< UpdateThreadData* >(threadarg);
    pthread_exit(static_cast< void* >(my_data));
}

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned, int num_threads)
{
    // Count number of data in each class
    std::vector< int > count(class_n);
    // Temporal point value to calculate distance
    std::vector< Point > ts(num_threads);
    std::vector< AssignThreadData > assignThreadDatas(num_threads, AssignThreadData(class_n, centroids, data, partitioned));
    int data_gap = static_cast< int >(std::ceil(data_n / num_threads));
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        AssignThreadData& my_data = assignThreadDatas[thread_id];
        my_data.t = &ts[thread_id];
        my_data.data_start_index = data_gap * thread_id;
        my_data.data_end_index = std::min(my_data.data_start_index + data_gap, data_n);
    }
    pthread_t thread[num_threads];
    pthread_attr_t attr;
    void* status;

    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // Iterate through number of interations
    for (int i = 0; i < iteration_n; i++) {
        // Assignment step
        for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            int rc = pthread_create(&thread[thread_id], &attr, AssignmentWork, static_cast< void* >(&assignThreadDatas[thread_id]));
            if (rc) {
                std::cout << "ERROR; return code from pthread_create() is " << rc << std::endl;
                exit(-1);
            }
        }

        pthread_attr_destroy(&attr);
        for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            int rc = pthread_join(thread[thread_id], &status);
            if (rc) {
                std::cout << "ERROR; return code from pthread_join() is " << rc << std::endl;
                exit(-1);
            }
        }

        // Update step
        // Clear sum buffer and class count
        for (int class_i = 0; class_i < class_n; class_i++) {
            centroids[class_i].x = 0.0;
            centroids[class_i].y = 0.0;
            count[class_i] = 0;
        }
        // Sum up and count data for each class
        for (int data_i = 0; data_i < data_n; data_i++) {
            centroids[partitioned[data_i]].x += data[data_i].x;
            centroids[partitioned[data_i]].y += data[data_i].y;
            count[partitioned[data_i]]++;
        }
        // Divide the sum with number of class for mean point
        for (int class_i = 0; class_i < class_n; class_i++) {
            centroids[class_i].x /= count[class_i];
            centroids[class_i].y /= count[class_i];
        }
    }
}
