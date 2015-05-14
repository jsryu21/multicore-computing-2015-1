__kernel void matrixmul(__global float* C, __global float* A, __global float* B, int wA, int wB) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    printf("%d %d\n", i, j);
};
