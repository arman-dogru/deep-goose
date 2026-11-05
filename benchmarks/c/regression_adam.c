// Same 2->1 batch regression and Adam update order as examples/regression/train.goose.
#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <math.h>
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
    enum { B = 256, EPOCHS = 250 };
    const float lr = 0.03f;
    float x[B*2], target[B];
    for (int r=0; r<B; ++r) {
        float f0 = ((float)(r % 17) - 8.0f) / 8.0f;
        float f1 = ((float)((r * 7) % 19) - 9.0f) / 9.0f;
        x[r*2] = f0; x[r*2+1] = f1;
        target[r] = 1.5f*f0 - 2.0f*f1 + 0.25f;
    }
    float w[2] = {0}, bias[1] = {0};
    float prediction[B], grad_y[B], grad_x[B*2], grad_w[2], grad_b[1];
    float first_w[2] = {0}, second_w[2] = {0};
    float first_b[1] = {0}, second_b[1] = {0};
    float beta1_power=1.0f, beta2_power=1.0f;
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
        beta1_power *= 0.9f;
        beta2_power *= 0.999f;
        const float correction1 = 1.0f-beta1_power;
        const float correction2 = 1.0f-beta2_power;
        for (int k=0; k<2; ++k) {
            const float g=grad_w[k];
            first_w[k] = 0.9f * first_w[k] + 0.1f * g;
            second_w[k] = 0.999f * second_w[k] + 0.001f * g * g;
            w[k] -= lr*(first_w[k]/correction1)/
                (sqrtf(second_w[k]/correction2)+1.0e-8f);
        }
        first_b[0] = 0.9f*first_b[0] + 0.1f*grad_b[0];
        second_b[0] = 0.999f*second_b[0] + 0.001f*grad_b[0]*grad_b[0];
        bias[0] -= lr*(first_b[0]/correction1)/
            (sqrtf(second_b[0]/correction2)+1.0e-8f);
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
    printf("ADAM_RESULT,c,%d,%" PRId64 ",%.9g,%.9g,%.9g,%.9g\n",
           EPOCHS, elapsed, loss, w[0], w[1], bias[0]);
    // Prevent the unused dx calculation from being removed by overly aggressive
    // optimization without making it part of training semantics.
    if (grad_x[0] == 1.0e30f) return 4;
    return 0;
}
