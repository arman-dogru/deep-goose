// Same 2->1 batch regression and SGD update order as examples/regression/train.goose.
#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int64_t ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) abort();
    return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}
int main(void) {
    enum { B = 256, EPOCHS = 200 };
    const float lr = 0.1f;
    float x[B*2], target[B];
    for (int r=0; r<B; ++r) {
        float f0 = ((float)(r % 17) - 8.0f) / 8.0f;
        float f1 = ((float)((r * 7) % 19) - 9.0f) / 9.0f;
        x[r*2] = f0; x[r*2+1] = f1;
        target[r] = 1.5f*f0 - 2.0f*f1 + 0.25f;
    }
    float w[2] = {0}, bias[1] = {0};
    float prediction[B], grad_y[B], grad_x[B*2], grad_w[2], grad_b[1];
    int64_t before = ns();
    for (int epoch=0; epoch<EPOCHS; ++epoch) {
        for (int r=0; r<B; ++r) {
            float s = bias[0];
            for (int k=0; k<2; ++k) s += x[r*2+k] * w[k];
            prediction[r] = s;
        }
        for (int r=0; r<B; ++r)
            grad_y[r] = 2.0f/(float)B * (prediction[r] - target[r]);
        for (int col=0; col<1; ++col) {
            float s = 0;
            for (int r=0; r<B; ++r) s += grad_y[r];
            grad_b[col] = s;
        }
        for (int k=0; k<2; ++k) {
            float s = 0;
            for (int r=0; r<B; ++r) s += x[r*2+k] * grad_y[r];
            grad_w[k] = s;
        }
        for (int r=0; r<B; ++r)
            for (int k=0; k<2; ++k) grad_x[r*2+k] = grad_y[r] * w[k];
        for (int k=0; k<2; ++k) w[k] -= lr * grad_w[k];
        bias[0] -= lr * grad_b[0];
    }
    int64_t elapsed = ns() - before;
    float loss = 0;
    for (int r=0; r<B; ++r) {
        float p = bias[0];
        for (int k=0; k<2; ++k) p += x[r*2+k] * w[k];
        float d = p-target[r];
        loss += d*d;
    }
    loss /= B;
    if (!(loss < 0.0001f)) return 3;
    printf("TRAIN_RESULT,c,%d,%" PRId64 ",%.9g,%.9g,%.9g,%.9g\n",
           EPOCHS, elapsed, loss, w[0], w[1], bias[0]);
    // Prevent the unused dx calculation from being removed by overly aggressive
    // optimization without making it part of training semantics.
    if (grad_x[0] == 1.0e30f) return 4;
    return 0;
}
