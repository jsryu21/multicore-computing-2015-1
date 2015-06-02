typedef struct PointType {
    float x; float y;
} Point;

__kernel void dispose_data(__global Point* centroids, __global Point* data, __global int* partitioned, __global Point* tempCentroids, __global int* tempCount, int data_n, int class_n, int num_threads) {
    int data_gap = ceil((float)data_n / num_threads);
    int thread_id = get_global_id(0);
    int data_start_index = data_gap * thread_id;
    int data_end_index = data_start_index + data_gap;
    if (data_end_index > data_n ) {
        data_end_index = data_n;
    }
    Point t;
    int data_i;
    int class_i;
    float min_dist;
    float dist;
    for (data_i = data_start_index; data_i < data_end_index; ++data_i) {
        min_dist = DBL_MAX;
        for (class_i = 0; class_i < class_n; class_i++) {
            t.x = data[data_i].x - centroids[class_i].x;
            t.y = data[data_i].y - centroids[class_i].y;
            dist = t.x * t.x + t.y * t.y;
            if (dist < min_dist) {
                partitioned[data_i] = class_i;
                min_dist = dist;
            }
        }
    }
    // Sum up and count data for each class
    for (data_i = data_start_index; data_i < data_end_index; data_i++) {
        tempCentroids[thread_id * class_n + partitioned[data_i]].x += data[data_i].x;
        tempCentroids[thread_id * class_n + partitioned[data_i]].y += data[data_i].y;
        tempCount[thread_id * class_n + partitioned[data_i]]++;
    }
};
