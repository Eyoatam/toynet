#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void copy_matrix(float *dest, float *src, int size) {
    for (int i = 0; i < size; i++) {
        dest[i] = src[i];
    }
}

void print_matrix(float *mat, int R, int C) {
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            printf("%f ", mat[i * C + j]);
        }
        printf("\n");
    }
}

// add a vector of shape (C,) to a matrix of shape (R, C)
void mat_add_vec(float *out, float *mat, float *vec, int R, int C) {
    copy_matrix(out, mat, R * C);
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            out[i * C + j] += vec[j];
        }
    }
}

void all_ones(float *mat, int R, int C) {
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            mat[i * C + j] = 1.0f;
        }
    }
}

void all_zeros(float *mat, int R, int C) {
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            mat[i * C + j] = 0.0f;
        }
    }
}

void matmul(float *out, float *mat1, float *mat2, int B, int X, int O) {
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < O; j++) {
            float value = 0.0f;
            for (int k = 0; k < X; k++) {
                value += mat1[i * X + k] * mat2[k * O + j];
            }
            out[i * O + j] = value;
        }
    }
}

float *matmul_element(float *mat1, float *mat2, int R, int C) {
    float *out = (float *)malloc(R * C * sizeof(float));
    assert(out != NULL);
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            out[i * C + j] = mat1[i * C + j] * mat2[i * C + j];
        }
    }
    return out;
}

void init_params(float *weight, float *bias, int R, int C) {
    srand((int)12345);
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            weight[i * C + j] = ((float)rand() / (float)(RAND_MAX)); // [0, RAND_MAX] -> [0, 1]
        }
    }
    for (int i = 0; i < C; i++) {
        bias[i] = ((float)rand() / (float)(RAND_MAX));
    }
}

float maxf(float a, float b) { return (a > b) ? a : b; }

float *relu(float *mat, int R, int C) {
    float *out = (float *)malloc(R * C * sizeof(float));
    assert(out != NULL);
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            out[i * C + j] = maxf(0.0f, mat[i * C + j]);
        }
    }
    return out;
}

float *relu_backward(float *mat, int R, int C) {
    float *out = (float *)malloc(R * C * sizeof(float));
    assert(out != NULL);
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            if (mat[i * C + j] < 0.0f) {
                out[i * C + j] = 0.0f;
            } else {
                out[i * C + j] = 1.0f;
            }
        }
    }
    return out;
}

float mean_squared_error(float *ytrue, float *ypred, int R, int C) {
    float error = 0.0f;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            error += powf(ytrue[i * C + j] - ypred[i * C + j], 2.0f);
        }
        error /= C;
    }
    return error;
}

float *mean_squared_error_backward(float *ytrue, float *ypred, int R, int C) {
    float *out = (float *)malloc(R * C * sizeof(float));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            out[i * C + j] = -2 * (ytrue[i * C + j] - ypred[i * C + j]);
        }
    }
    return out;
}

typedef struct {
    float *weight;
    float *bias;
    float *wx;
    float *wx_b;
    float *fwx_b;
} Dense;

Dense dense_layer(float *input, int batch_size, int input_size, int output_size,
                  float *(*activation)(float *, int, int), float *weight, float *bias) {
    // dot(x,w)
    float *wx = (float *)malloc(batch_size * output_size * sizeof(float));
    assert(wx != NULL);
    matmul(wx, input, weight, batch_size, input_size, output_size);
    // dot(x, w) + b
    float *wx_b = (float *)malloc(batch_size * output_size * sizeof(float));
    assert(wx_b != NULL);
    mat_add_vec(wx_b, wx, bias, batch_size, output_size);
    // activation(dot(x, w) + b)
    float *fwx_b = activation(wx_b, batch_size, output_size);
    return (Dense){weight, bias, wx, wx_b, fwx_b};
}

void update_param(float *param, float *gradient, int R, int C, float lr) {
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            param[i * C + j] -= lr * gradient[i * C + j];
        }
    }
}

typedef struct {
    float *weight;
    float *bias;
} Layer;

int main() {
    float input[9] = {
        2.0, 3.0, 1.0, //
        3.0, 1.0, 1.0, //
        1.0, 1.0, 1.0,
    };
    float true_y[9] = {
        1.0, 3.0, 2.0, //
        1.0, 2.0, 2.0, //
        2.0, 2.0, 2.0,
    };

    // init params
    Layer layer1 = {.weight = (float *)malloc(3 * 3 * sizeof(float)),
                    .bias = (float *)malloc(3 * sizeof(float))};
    assert(layer1.weight != NULL);
    assert(layer1.bias != NULL);
    Layer layer2 = {.weight = (float *)malloc(3 * 3 * sizeof(float)),
                    .bias = (float *)malloc(3 * sizeof(float))};
    assert(layer1.weight != NULL);
    assert(layer1.bias != NULL);
    init_params(layer1.weight, layer1.bias, 3, 3);
    init_params(layer2.weight, layer2.bias, 3, 3);

    for (int i = 0; i < 50; i++) {
        // forward pass
        Dense dense_1 = dense_layer(input, 3, 3, 3, relu, layer1.weight, layer1.bias);
        Dense dense_2 = dense_layer(dense_1.fwx_b, 3, 3, 3, relu, layer2.weight, layer2.bias);

        // backward pass
        float *dl_df = mean_squared_error_backward(true_y, dense_2.fwx_b, 3, 3);
        float *df_dwxb2 = relu_backward(dense_2.fwx_b, 3, 3);
        float *dwxb2_dwx = (float *)malloc(3 * 3 * sizeof(float));
        all_ones(dwxb2_dwx, 3, 3);
        float *dwxb2_db2 = (float *)malloc(3 * 3 * sizeof(float));
        all_ones(dwxb2_db2, 3, 3);
        float *dwx_dw2 = dense_1.fwx_b;
        float *tmp_1 = matmul_element(matmul_element(dl_df, df_dwxb2, 3, 3), dwxb2_dwx, 3, 3);
        float *dl_dw2 = matmul_element(tmp_1, dwx_dw2, 3, 3);
        float *dl_db2 = matmul_element(matmul_element(dl_df, df_dwxb2, 3, 3), dwxb2_db2, 3, 3);

        float learning_rate = 1e-3;
        update_param(layer2.weight, dl_dw2, 3, 3, learning_rate);
        update_param(layer2.bias, dl_db2, 3, 3, learning_rate);

        float *dwx_df1 = layer1.weight;
        float *df1_dwxb = relu_backward(dense_1.fwx_b, 3, 3);
        float *dwxb_dwx = (float *)malloc(3 * 3 * sizeof(float));
        all_ones(dwxb_dwx, 3, 3);
        float *dwxb_db1 = (float *)malloc(3 * 3 * sizeof(float));
        all_ones(dwxb_db1, 3, 3);
        float *dwx_dw1 = input;
        // dl_df x df_dwxb2 x dwxb2_dwx x dwx_df1 x df1_dwxb x dwxb_dwx x dwx_dw1
        float *tmp_2 = matmul_element(matmul_element(tmp_1, dwx_df1, 3, 3), df1_dwxb, 3, 3);
        float *dl_dw1 = matmul_element(matmul_element(tmp_2, dwxb_dwx, 3, 3), dwx_dw1, 3, 3);
        // dl_df x df_dwxb2 x dwxb2_dwx x dwx_df1 x df1_dwxb x dwxb_db1
        float *dl_db1 = matmul_element(tmp_2, dwxb_db1, 3, 3);

        update_param(layer1.weight, dl_dw1, 3, 3, learning_rate);
        update_param(layer1.bias, dl_db1, 3, 3, learning_rate);

        float loss = mean_squared_error(true_y, dense_2.fwx_b, 3, 3);
        printf("loss: %f. iteration: %d\n", loss, i);

        free(dl_df);
        free(df_dwxb2);
        free(dwxb2_dwx);
        free(dwxb2_db2);
        free(tmp_1);
        free(dl_dw2);
        free(dl_db2);
        free(df1_dwxb);
        free(dwxb_dwx);
        free(dwxb_db1);
        free(tmp_2);
        free(dl_dw1);
        free(dl_db1);
        free(dense_1.fwx_b);
        free(dense_1.wx);
        free(dense_1.wx_b);
        free(dense_2.fwx_b);
        free(dense_2.wx);
        free(dense_2.wx_b);
    }

    // predictions
    Dense dense_1 = dense_layer(input, 3, 3, 3, relu, layer1.weight, layer1.bias);
    Dense dense_2 = dense_layer(dense_1.fwx_b, 3, 3, 3, relu, layer2.weight, layer2.bias);
    printf("predictions: \n");
    print_matrix(dense_2.fwx_b, 3, 3);

    free(layer1.weight);
    free(layer1.bias);
    free(layer2.weight);
    free(layer2.bias);
    return 0;
}
