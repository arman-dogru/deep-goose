// Identical synthetic dense forward benchmark in C, compiled with Clang -O2.
// Usage: dense_study <naive|rows> <batch> <in_features> <out_features> <iterations>
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int64_t ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) abort();
    return (int64_t)t.tv_sec*1000000000LL + (int64_t)t.tv_nsec;
}
static int parse(const char *s, int max) {
    char *end=NULL;
    errno=0;
    long v=strtol(s,&end,10);
    if(errno || end==s || *end || v<1 || v>max) {
        fprintf(stderr,"invalid parameter: %s\n",s);
        exit(2);
    }
    return (int)v;
}
static void dense_naive(const float *x, const float *w, const float *bias,
                        float *y, size_t batch, size_t inner, size_t outer) {
    for(size_t i=0; i<batch; ++i)
        for(size_t j=0; j<outer; ++j) {
            float sum=bias[j];
            for(size_t k=0; k<inner; ++k)
                sum+=x[i*inner+k]*w[k*outer+j];
            y[i*outer+j]=sum;
        }
}
static void dense_rows(const float *x, const float *w, const float *bias,
                       float *y, size_t batch, size_t inner, size_t outer) {
    for(size_t i=0; i<batch; ++i) {
        const float *xrow=x+i*inner;
        float *yrow=y+i*outer;
        for(size_t j=0;j<outer;++j) yrow[j]=bias[j];
        for(size_t k=0;k<inner;++k) {
            const float *wrow=w+k*outer;
            const float value=xrow[k];
            for(size_t j=0;j<outer;++j) yrow[j]+=value*wrow[j];
        }
    }
}
int main(int argc, char **argv) {
    if(argc!=6 || (strcmp(argv[1],"naive") && strcmp(argv[1],"rows"))) {
        fprintf(stderr,"usage: dense_study <naive|rows> <batch> <in> <out> <iterations>\n");
        return 2;
    }
    int B=parse(argv[2],1024), K=parse(argv[3],512), N=parse(argv[4],512);
    int iterations=parse(argv[5],10000);
    float *x=malloc((size_t)B*K*sizeof(float));
    float *w=malloc((size_t)K*N*sizeof(float));
    float *bias=malloc((size_t)N*sizeof(float));
    float *y=malloc((size_t)B*N*sizeof(float));
    if(!x||!w||!bias||!y) {free(x);free(w);free(bias);free(y);return 2;}
    for(int i=0;i<B*K;++i) x[i]=((float)((i*13)%211)-105.0f)*0.00317f;
    for(int i=0;i<K*N;++i) w[i]=((float)((i*23)%191)-95.0f)*0.00279f;
    for(int i=0;i<N;++i) bias[i]=((float)((i*7)%53)-26.0f)*0.00831f;
    for(int i=0;i<B*N;++i) y[i]=0.0f;
    const int naive=!strcmp(argv[1],"naive");
    if(naive) dense_naive(x,w,bias,y,B,K,N);
    else dense_rows(x,w,bias,y,B,K,N);
    int64_t before=ns();
    for(int iteration=0;iteration<iterations;++iteration) {
        if(naive) dense_naive(x,w,bias,y,B,K,N);
        else dense_rows(x,w,bias,y,B,K,N);
    }
    int64_t elapsed=ns()-before;
    double max_error=0,checksum=0;
    for(int i=0;i<B;++i) {
        for(int j=0;j<N;++j) {
            double reference=(double)bias[j];
            for(int k=0;k<K;++k)
                reference+=(double)x[i*K+k]*(double)w[k*N+j];
            double actual=(double)y[i*N+j],error=fabs(actual-reference);
            if(error>0.0005+0.0001*fabs(reference)) {
                fprintf(stderr,"dense mismatch i=%d j=%d error=%g\n",i,j,error);
                free(x);free(w);free(bias);free(y);
                return 3;
            }
            if(error>max_error)max_error=error;
            checksum+=actual;
        }
    }
    printf("DENSE_RESULT,c,%s,%d,%d,%d,%d,%" PRId64 ",%.17g,%.17g\n",
           argv[1],B,K,N,iterations,elapsed,max_error,checksum);
    free(x);free(w);free(bias);free(y);
    return 0;
}
