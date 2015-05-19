__kernel void matrixmul(__global float* C, __global float* A, __global float* B, int wA, int wB, int wC, int hC, int global_i, int global_j) {
    int i_gap = ceil((float)wC / global_i);
    int i = get_global_id(0);
    int i_start_index = i_gap * i;
    int i_end_index = i_start_index + i_gap;
    if (i_end_index > wC) {
        i_end_index = wC;
    }
    int j_gap = ceil((float)hC / global_j);
    int j = get_global_id(1);
    int j_start_index = j_gap * j;
    int j_end_index = j_start_index + j_gap;
    if (j_end_index > hC) {
        j_end_index = hC;
    }
    //printf("%d %d\n", i, j);
    int k;
    for (i = i_start_index; i < i_end_index; ++i) {
        for (j = j_start_index; j < j_end_index; ++j) {
            float acc = 0.f;
            for (k = 0; k < wA; ++k) {
                acc += A[j * wA + k] * B[k * wB + i];
            }
            C[j * wC + i] = acc;
        }
    }
};
