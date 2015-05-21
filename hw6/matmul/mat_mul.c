#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include "timers.h"

int NDIM = 2048;
#define MIN(a,b) (((a)<(b))?(a):(b))
int NUM_THREADS = 4;

float** a;
float** b;
float** c;

int print_matrix = 0;
int validation = 0;

void mat_mul( float** c, float** a, float** b )
{
    int i, j, k;
#pragma omp parallel for
    for( i = 0; i < NDIM; i++ )
    {
        for( j = 0; j < NDIM; j++ )
        {
            for( k = 0; k < NDIM; k++ )
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
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
    printf("  -t 4 : designate the number of threads(default : 4).\n");
    printf("  -m 2048 : designate the number of matrix size(default : 2048).\n");
}

void parse_opt(int argc, char** argv)
{
    int opt;

    while( (opt = getopt(argc, argv, "pvht:m:ikjs:")) != -1 )
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
                NUM_THREADS = atoi(optarg);
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
    int i, j, k = 1;

    parse_opt( argc, argv );
    a = (float**)malloc(NDIM * sizeof(float*));
    for (i = 0; i < NDIM; ++i) {
        a[i] = (float*)malloc(NDIM * sizeof(float));
    }
    b = (float**)malloc(NDIM * sizeof(float*));
    for (i = 0; i < NDIM; ++i) {
        b[i] = (float*)malloc(NDIM * sizeof(float));
    }
    c = (float**)malloc(NDIM * sizeof(float*));
    for (i = 0; i < NDIM; ++i) {
        c[i] = (float*)malloc(NDIM * sizeof(float));
    }

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
