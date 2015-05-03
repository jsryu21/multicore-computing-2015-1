
#include "kmeans.h"
#include <cstdlib>
#include <cfloat>
#include <pthread.h>
#include <vector>
#include <cmath>
#include <iostream>

class ILock {
    public:
        virtual void Lock() = 0;
        virtual void Unlock() = 0;
};

class TTASlock : public ILock {
    public:
        TTASlock()
            : state(false)
        {
        }
        void Lock()
        {
            while (true)
            {
                while (state) {}
                if (__sync_lock_test_and_set(&state, true) == false)
                {
                    return;
                }
            }
        }
        void Unlock()
        {
            __sync_lock_release(&state);
        }
    private:
        bool state;
};

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
    UpdateThreadData(const Point* data, const int* partitioned, ILock& lock_, int class_n, Point* centroids, std::vector< int >& count_);
    const Point* data;
    const int* partitioned;
    ILock& lock;
    int class_n;
    Point* centroids;
    std::vector< int >& count;
    int data_start_index;
    int data_end_index;
    std::vector< Point >* temp_centroids;
    std::vector< int >* temp_count;
};

UpdateThreadData::UpdateThreadData(const Point* data, const int* partitioned, ILock& lock_, int class_n, Point* centroids, std::vector< int >& count_) : lock(lock_), count(count_)
{
    this->data = data;
    this->partitioned = partitioned;
    this->lock = lock;
    this->class_n = class_n;
    this->centroids = centroids;
}

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
    const Point* data = my_data->data;
    const int* partitioned = my_data->partitioned;
    ILock& lock = my_data->lock;
    int class_n = my_data->class_n;
    Point* centroids = my_data->centroids;
    std::vector< int >& count = my_data->count;
    int data_start_index = my_data->data_start_index;
    int data_end_index = my_data->data_end_index;
    std::vector< Point >* temp_centroids = my_data->temp_centroids;
    std::vector< int >* temp_count = my_data->temp_count;

    // Sum up and count data for each class
    for (int data_i = data_start_index; data_i < data_end_index; data_i++) {
        (*temp_centroids)[partitioned[data_i]].x += data[data_i].x;
        (*temp_centroids)[partitioned[data_i]].y += data[data_i].y;
        (*temp_count)[partitioned[data_i]]++;
    }

    lock.Lock();
    for (int class_i = 0; class_i < class_n; ++class_i) {
        centroids[class_i].x += (*temp_centroids)[class_i].x;
        centroids[class_i].y += (*temp_centroids)[class_i].y;
        count[class_i] += (*temp_count)[class_i];
    }
    lock.Unlock();

    pthread_exit(static_cast< void* >(my_data));
}

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned, int num_threads)
{
    // Count number of data in each class
    std::vector< int > count(class_n);
    // Temporal point value to calculate distance
    std::vector< Point > ts(num_threads, Point(0.f, 0.f));
    std::vector< AssignThreadData > assignThreadDatas(num_threads, AssignThreadData(class_n, centroids, data, partitioned));
    int data_gap = static_cast< int >(std::ceil(data_n / num_threads));
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        AssignThreadData& my_data = assignThreadDatas[thread_id];
        my_data.t = &ts[thread_id];
        my_data.data_start_index = data_gap * thread_id;
        my_data.data_end_index = std::min(my_data.data_start_index + data_gap, data_n);
    }
    TTASlock lock;
    std::vector< std::vector< Point > > tempCentroids(num_threads, std::vector< Point >(class_n, Point(0.f, 0.f)));
    std::vector< std::vector< int > > tempCount(num_threads, std::vector< int >(class_n));
    std::vector< UpdateThreadData > updateThreadDatas(num_threads, UpdateThreadData(data, partitioned, lock, class_n, centroids, count));
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        UpdateThreadData& my_data = updateThreadDatas[thread_id];
        my_data.data_start_index = data_gap * thread_id;
        my_data.data_end_index = std::min(my_data.data_start_index + data_gap, data_n);
        my_data.temp_centroids = &tempCentroids[thread_id];
        my_data.temp_count = &tempCount[thread_id];
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
        std::fill(centroids, centroids + class_n, Point(0.f, 0.f));
        std::fill(count.begin(), count.end(), 0);
        std::fill(tempCentroids.begin(), tempCentroids.end(), std::vector< Point >(class_n, Point(0.f, 0.f)));
        std::fill(tempCount.begin(), tempCount.end(), std::vector< int >(class_n, 0));

        for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            int rc = pthread_create(&thread[thread_id], &attr, UpdateWork, static_cast< void* >(&updateThreadDatas[thread_id]));
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
        // Divide the sum with number of class for mean point
        for (int class_i = 0; class_i < class_n; class_i++) {
            centroids[class_i].x /= count[class_i];
            centroids[class_i].y /= count[class_i];
        }
    }
}
