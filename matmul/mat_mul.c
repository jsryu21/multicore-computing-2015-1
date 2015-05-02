#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "timers.h"

#define NDIM    2048
#define MIN(a,b) (((a)<(b))?(a):(b))
int NUM_THREADS = 4;

float a[NDIM][NDIM];
float b[NDIM][NDIM];
float c[NDIM][NDIM];

int print_matrix = 0;
int validation = 0;

struct thread_data {
    int thread_id;
    int from;
    int to;
};

struct thread_data* thread_data_array;

void *BusyWork(void *threadarg)
{
    struct thread_data* my_data = (struct thread_data*)threadarg;
    int i, j, k;
    for (k = 0; k < NDIM; ++k) {
        for (i = my_data->from; i < my_data->to; ++i) {
            float r = a[i][k];
            for (j = 0; j < NDIM; ++j) {
                c[i][j] += r * b[k][j];
            }
        }
    }
    pthread_exit((void*)my_data);
}

void mat_mul( float c[NDIM][NDIM], float a[NDIM][NDIM], float b[NDIM][NDIM] )
{
    /* https://computing.llnl.gov/tutorials/pthreads/#Joining */
    pthread_t thread[NUM_THREADS];
    pthread_attr_t attr;
    int rc;
    int t;
    void* status;
    int gap = (int)ceil(NDIM / NUM_THREADS);

    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for(t = 0; t < NUM_THREADS; t++) {
        struct thread_data* my_data = &thread_data_array[t];
        my_data->thread_id = t;
        my_data->from = gap * t;
        my_data->to = MIN(my_data->from + gap, NDIM);
        rc = pthread_create(&thread[t], &attr, BusyWork, (void*)my_data);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    /* Free attribute and wait for the other threads */
    pthread_attr_destroy(&attr);
    for(t = 0; t < NUM_THREADS; t++) {
        rc = pthread_join(thread[t], &status);
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }
}

/************************** DO NOT TOUCH BELOW HERE ******************************/

void check_mat_mul( float c[NDIM][NDIM], float a[NDIM][NDIM], float b[NDIM][NDIM] )
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

void print_mat( float mat[NDIM][NDIM] )
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
	printf("  -t 4 : designate the number of threads(default : 4).\n");
}

void parse_opt(int argc, char** argv)
{
	int opt;

	while( (opt = getopt(argc, argv, "pvht:ikjs:")) != -1 )
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

        case 't':
            NUM_THREADS = *optarg - '0';
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
	int i, j, k = 1;

	parse_opt( argc, argv );
    thread_data_array = (struct thread_data*)malloc(sizeof(struct thread_data) * NUM_THREADS);

	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			a[i][j] = k;
			b[i][j] = k;
			k++;
		}
	}

	timer_start(1);
	mat_mul( c, a, b );
	timer_stop(1);

	printf("Time elapsed : %lf sec\n", timer_read(1));


	if( validation )
		check_mat_mul( c, a, b );

	if( print_matrix )
	{
		printf("MATRIX A: \n");
		print_mat(a);

		printf("MATRIX B: \n");
		print_mat(b);

		printf("MATRIX C: \n");
		print_mat(c);
	}

    free(thread_data_array);
	return 0;
}
