#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include "timers.h"
#include "mpi.h"
#include <string.h>

int NDIM = 2048;
#define MIN(a,b) (((a)<(b))?(a):(b))

float** a;
float** b;
float** c;

int print_matrix = 0;
int validation = 0;

int FROM_MASTER = 1;
int FROM_WORKER = 2;

int mat_mul( float** c, float** a, float** b, int argc, char** argv )
{
    int numtasks, taskid;
    int gap, remain, task_i, from, to, row_i;
    int i, j, r;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Status status;

    if (taskid == 0) {
        float* temp_c = (float*)malloc(NDIM * sizeof(float));

        // Initialize matrix a, b
        for( i = 0; i < NDIM; i++ ) {
            for( j = 0; j < NDIM; j++ ) {
                a[i][j] = i * j;
                b[i][j] = i + j;
            }
        }

        // Scatter matrix a
        gap = NDIM / (numtasks - 1);
        remain = NDIM - gap * (numtasks - 1);
        from = 0;
        to = 0;
        for (task_i = 1; task_i < numtasks; ++task_i) {
            from = to;
            to = from + gap;
            if (remain > 0) {
                to++;
                remain--;
            }
            for (row_i = from; row_i < to; ++row_i) {
                MPI_Send(a[row_i], NDIM, MPI_FLOAT, task_i, row_i, MPI_COMM_WORLD);
            }
        }

        // Broadcast matrix b
        for (i = 0; i < NDIM; ++i) {
            MPI_Bcast(b[i], NDIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // Receive reducing result matrix c
        for (i = 0; i < NDIM; ++i) {
            MPI_Recv(temp_c, NDIM, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            memcpy(c[status.MPI_TAG], temp_c, NDIM * sizeof(float));
        }

        free(temp_c);
    } else {
        float* temp_a = (float*)malloc(NDIM * sizeof(float));

        // Initialize matrix a, b
        for (i = 0; i < NDIM; ++i) {
            memset(a[i], 0, NDIM * sizeof(float));
            memset(b[i], 0, NDIM * sizeof(float));
        }

        // Receive scattered matrix a
        gap = NDIM / (numtasks - 1);
        remain = NDIM - gap * (numtasks - 1);
        from = 0;
        to = 0;
        for (task_i = 1; task_i < numtasks; ++task_i) {
            from = to;
            to = from + gap;
            if (remain > 0) {
                to++;
                remain--;
            }
            if (task_i == taskid) {
                break;
            }
        }
        for (row_i = from; row_i < to; ++row_i) {
            MPI_Recv(temp_a, NDIM, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            memcpy(a[status.MPI_TAG], temp_a, NDIM * sizeof(float));
        }

        // Receive broadcasted matrix b
        for (i = 0; i < NDIM; ++i) {
            MPI_Bcast(b[i], NDIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        for (j = 0; j < NDIM; ++j) {
            for (row_i = from; row_i < to; ++row_i) {
                r = a[row_i][j];
                for (i = 0; i < NDIM; ++i) {
                    c[row_i][i] += r * b[j][i];
                }
            }
        }

        // Reduce result matrix c
        for (row_i = from; row_i < to; ++row_i) {
            MPI_Send(c[row_i], NDIM, MPI_FLOAT, 0, row_i, MPI_COMM_WORLD);
        }

        free(temp_a);
    }

    MPI_Finalize();
    return taskid;
}

/************************** DO NOT TOUCH BELOW HERE ******************************/

void check_mat_mul( float** c, float** a, float** b )
{
    int i, j, k;
    float sum;
    int validated = 1;

    printf("Validating the result..\n");

    // C = AB
    for( i = 0; i < NDIM; i++ )
    {
        for( j = 0; j < NDIM; j++ )
        {
            sum = 0;
            for( k = 0; k < NDIM; k++ )
            {
                sum += a[i][k] * b[k][j];
            }

            if( c[i][j] != sum )
            {
                printf("c[%d][%d] is differ(value=%lf correct_value=%lf)!!\n", i, j, c[i][j], sum );
                validated = 0;
            }
        }
    }

    printf("Validation : ");
    if( validated )
        printf("SUCCESSFUL.\n");
    else
        printf("FAILED.\n");
}

void print_mat( float** mat )
{
    int i, j;

    for( i = 0; i < NDIM; i++ )
    {
        for( j = 0; j < NDIM; j++ )
        {
            printf("%8.2lf ", mat[i][j]);
        }
        printf("\n");
    }
}

void print_help(const char* prog_name)
{
    printf("Usage: %s [-pvht]\n", prog_name );
    printf("\n");
    printf("OPTIONS\n");
    printf("  -p : print matrix data.\n");
    printf("  -v : validate matrix multiplication.\n");
    printf("  -h : print this page.\n");
    printf("  -m 2048 : designate the number of matrix size(default : 2048).\n");
}

void parse_opt(int argc, char** argv)
{
    int opt;

    while( (opt = getopt(argc, argv, "pvhm:")) != -1 )
    {
        switch(opt)
        {
            case 'p':
                // print matrix data.
                print_matrix = 1;
                break;

            case 'v':
                // validation
                validation = 1;
                break;

            case 'm':
                NDIM = atoi(optarg);
                break;

            case 'h':
            default:
                print_help(argv[0]);
                exit(0);
                break;
        }
    }
}

int main(int argc, char** argv)
{
    int i;
    int ret;

    parse_opt( argc, argv );
    a = (float**)malloc(NDIM * sizeof(float*));
    for (i = 0; i < NDIM; ++i) {
        a[i] = (float*)malloc(NDIM * sizeof(float));
    }
    b = (float**)malloc(NDIM * sizeof(float*));
    for (i = 0; i < NDIM; ++i) {
        b[i] = (float*)malloc(NDIM * sizeof(float));
    }
    c = (float**)calloc(NDIM, sizeof(float*));
    for (i = 0; i < NDIM; ++i) {
        c[i] = (float*)calloc(NDIM, sizeof(float));
    }

    timer_start(1);
    ret = mat_mul( c, a, b, argc, argv );
    timer_stop(1);

    printf("Time elapsed : %lf sec\n", timer_read(1));


    if( ret == 0 && validation )
        check_mat_mul( c, a, b );

    if( ret == 0 && print_matrix )
    {
        printf("MATRIX A: \n");
        print_mat(a);

        printf("MATRIX B: \n");
        print_mat(b);

        printf("MATRIX C: \n");
        print_mat(c);
    }

    for (i = 0; i < NDIM; ++i) {
        free(c[i]);
    }
    free(c);
    for (i = 0; i < NDIM; ++i) {
        free(b[i]);
    }
    free(b);
    for (i = 0; i < NDIM; ++i) {
        free(a[i]);
    }
    free(a);
    return 0;
}
