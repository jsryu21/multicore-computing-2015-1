#include "kmeans.h"
#include "mpi.h"
#include <cfloat>

void AddPoint(Point* lhs, Point* rhs, int* len, MPI_Datatype* dptr) {
    for (int i = 0; i < *len; ++i) {
        rhs->x += lhs->x;
        rhs->y += lhs->y;
        lhs++;
        rhs++;
    }
}

int kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned, int num_threads, int local_size, int argc, char** argv)
{
    int numnodes, myid;
    int mpi_err = MPI_Init(&argc, &argv);
    mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &numnodes);
    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Status status;

    // https://www.msi.umn.edu/workshops/mpi/hands-on/derived-datatypes/struct/assign
    MPI_Datatype mystruct;
    int          blocklens[2];
    MPI_Aint     indices[2];
    MPI_Datatype old_types[2];

    /* One value of each type */
    blocklens[0] = 1;
    blocklens[1] = 1;
    /* The base types */
    old_types[0] = MPI_FLOAT;
    old_types[1] = MPI_FLOAT;
    /* The locations of each element */
    MPI_Address( &centroids[0].x, &indices[0] );
    MPI_Address( &centroids[0].y, &indices[1] );
    /* Make relative */
    indices[1] = indices[1] - indices[0];
    indices[0] = 0;
    MPI_Type_struct( 2, blocklens, indices, old_types, &mystruct );
    MPI_Type_commit( &mystruct );

    MPI_Op myOp;
    MPI_Op_create((MPI_User_function*)AddPoint, true, &myOp);

    // Count number of data in each class
    int* count = new int[class_n];
    Point* tempCentroids = new Point[class_n];
    int* tempCount = new int[class_n];

    int from;
    int to = 0;

    int gap = data_n / numnodes;
    int remain = data_n - gap * numnodes;
    for (int task_i = 0; task_i < numnodes; ++task_i) {
        from = to;
        to = from + gap;
        if (remain > 0) {
            to++;
            remain--;
        }

        if (task_i == myid) {
            break;
        }
    }

    printf("myid : %d, from : %d, to : %d\n", myid, from, to);

    // Iterate through number of interations
    for (int i = 0; i < iteration_n; i++) {
        memset(tempCentroids, 0, class_n * sizeof(Point));
        memset(tempCount, 0, class_n * sizeof(int));

        for (int class_i = 0; class_i < class_n; ++class_i) {
            MPI_Bcast(&centroids[class_i], 1, mystruct, 0, MPI_COMM_WORLD);
        }

        // Assignment step
        for (int data_i = from; data_i < to; ++data_i) {
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
            int index = partitioned[data_i];
            tempCentroids[index].x += data[data_i].x;
            tempCentroids[index].y += data[data_i].y;
            tempCount[index]++;
        }

        // Update step
        if (myid == 0) {
            // Clear sum buffer and class count
            memset(centroids, 0, class_n * sizeof(Point));
            memset(count, 0, class_n * sizeof(int));
        }

        for (int class_i = 0; class_i < class_n; ++class_i) {
            MPI_Reduce(&tempCentroids[class_i], &centroids[class_i], 1, mystruct, myOp, 0, MPI_COMM_WORLD);
        }
        MPI_Reduce(tempCount, count, class_n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (myid == 0) {
            for (int class_i = 0; class_i < class_n; ++class_i) {
                // Divide the sum with number of class for mean point
                centroids[class_i].x /= count[class_i];
                centroids[class_i].y /= count[class_i];
            }
        }
    }

    if (myid == 0) {
        int* temp = new int[data_n];
        memcpy(temp, partitioned, data_n);
        int gap = data_n / numnodes;
        int remain = data_n - gap * numnodes;
        for (int task_i = 1; task_i < numnodes; ++task_i) {
            from = to;
            to = from + gap;
            if (remain > 0) {
                to++;
                remain--;
            }
            MPI_Recv(&temp[from], to - from, MPI_INT, task_i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
        memcpy(partitioned, temp, data_n);
        delete temp;
    } else {
        MPI_Send(&partitioned[from], to - from, MPI_INT, 0, from, MPI_COMM_WORLD);
        //MPI_Gather(&partitioned[from], to - from, MPI_INT, NULL, to - from, MPI_INT, 0, MPI_COMM_WORLD);
    }

    delete[] tempCount;
    delete[] tempCentroids;
    delete[] count;

    MPI_Type_free(&mystruct);
    MPI_Finalize();
    return myid;
}
