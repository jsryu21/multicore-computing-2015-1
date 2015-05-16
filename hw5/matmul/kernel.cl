__kernel void matrixmul(__global float* C, __global float* A, __global float* B, int wA, int wB) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    //printf("%d %d\n", i, j);
    int k;
    float acc = 0.f;
    for (k = 0; k < wA; ++k) {
        acc += A[j * wA + k] * B[k * wB + i];
    }
    C[j * wB + i] = acc;
};
