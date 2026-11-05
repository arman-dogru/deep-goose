// Matched C control for Goose dense backward, with independent f64 references.
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int64_t clock_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t) != 0) abort();
    return (int64_t)t.tv_sec*1000000000LL+t.tv_nsec;
}
static long argnum(const char *src,long maximum) {
    char *end=NULL;
    errno=0;
    long n=strtol(src,&end,10);
    if(errno || end==src || *end || n<1 || n>maximum) {
        fprintf(stderr,"invalid positive integer: %s\n",src);exit(2);
    }
    return n;
}
static void backward_naive(const float *x,const float *w,const float *dy,
                           float *dx,float *dw,float *db,
                           size_t batch,size_t inner,size_t outer) {
    for(size_t col=0;col<outer;++col) {
        float s=0;
        for(size_t row=0;row<batch;++row) s+=dy[row*outer+col];
        db[col]=s;
    }
    for(size_t k=0;k<inner;++k) {
        for(size_t col=0;col<outer;++col) {
            float s=0;
            for(size_t row=0;row<batch;++row)
                s+=x[row*inner+k]*dy[row*outer+col];
            dw[k*outer+col]=s;
        }
    }
    for(size_t row=0;row<batch;++row) {
        for(size_t k=0;k<inner;++k) {
            float s=0;
            for(size_t col=0;col<outer;++col)
                s+=dy[row*outer+col]*w[k*outer+col];
            dx[row*inner+k]=s;
        }
    }
}
static void backward_dw_rows(const float *x,const float *dy,
                             float *dw,float *db,
                             size_t batch,size_t inner,size_t outer) {
    for(size_t col=0;col<outer;++col) db[col]=0;
    for(size_t idx=0;idx<inner*outer;++idx) dw[idx]=0;
    for(size_t row=0;row<batch;++row) {
        const float *xr=x+row*inner;
        const float *dyr=dy+row*outer;
        for(size_t col=0;col<outer;++col) db[col]+=dyr[col];
        for(size_t k=0;k<inner;++k) {
            float *dwr=dw+k*outer;
            const float xv=xr[k];
            for(size_t col=0;col<outer;++col)
                dwr[col]+=xv*dyr[col];
        }
    }
}
static void backward_dx_naive(const float *w,const float *dy,float *dx,
                              size_t batch,size_t inner,size_t outer) {
    for(size_t row=0;row<batch;++row)
        for(size_t k=0;k<inner;++k) {
            float s=0;
            for(size_t col=0;col<outer;++col)
                s+=dy[row*outer+col]*w[k*outer+col];
            dx[row*inner+k]=s;
        }
}
static void backward_dx_rows(const float *w,const float *dy,float *dx,
                             size_t batch,size_t inner,size_t outer) {
    for(size_t row=0;row<batch;++row) {
        const float *dyr=dy+row*outer;
        float *dxr=dx+row*inner;
        for(size_t k=0;k<inner;++k) {
            const float *wr=w+k*outer;
            float s=0;
            for(size_t col=0;col<outer;++col)
                s+=dyr[col]*wr[col];
            dxr[k]=s;
        }
    }
}
static void perform(const char *variant,
                    const float *x,const float *w,const float *dy,
                    float *dx,float *dw,float *db,
                    size_t batch,size_t inner,size_t outer) {
    if(!strcmp(variant,"naive")) {
        backward_naive(x,w,dy,dx,dw,db,batch,inner,outer);
    } else {
        backward_dw_rows(x,dy,dw,db,batch,inner,outer);
        if(!strcmp(variant,"rows_dw"))
            backward_dx_naive(w,dy,dx,batch,inner,outer);
        else
            backward_dx_rows(w,dy,dx,batch,inner,outer);
    }
}
static void validate(double actual,double reference,double *max_error,double *checksum,
                     const char *which,size_t i,size_t j) {
    double error=fabs(actual-reference);
    if(error>0.0005+0.0001*fabs(reference)) {
        fprintf(stderr,"%s mismatch [%zu,%zu] actual=%.12g expected=%.12g error=%.12g\n",
                which,i,j,actual,reference,error);
        exit(3);
    }
    if(error>*max_error)*max_error=error;
    *checksum+=actual;
}
int main(int argc,char **argv) {
    if(argc!=6 ||
       (strcmp(argv[1],"naive") && strcmp(argv[1],"rows_dw") && strcmp(argv[1],"rows_all"))) {
        fprintf(stderr,"usage: backward_study <naive|rows_dw|rows_all> B K N repetitions\n");
        return 2;
    }
    const char *variant=argv[1];
    const size_t B=(size_t)argnum(argv[2],512);
    const size_t K=(size_t)argnum(argv[3],512);
    const size_t N=(size_t)argnum(argv[4],512);
    const long iterations=argnum(argv[5],10000);
    float *x=malloc(B*K*sizeof(float)), *w=malloc(K*N*sizeof(float));
    float *dy=malloc(B*N*sizeof(float)), *dx=malloc(B*K*sizeof(float));
    float *dw=malloc(K*N*sizeof(float)), *db=malloc(N*sizeof(float));
    if(!x||!w||!dy||!dx||!dw||!db) {fprintf(stderr,"alloc failed\n");return 2;}
    for(size_t i=0;i<B*K;++i) {x[i]=((float)((i*13)%211)-105.0f)*0.00317f;dx[i]=0;}
    for(size_t i=0;i<K*N;++i) {w[i]=((float)((i*23)%191)-95.0f)*0.00279f;dw[i]=0;}
    for(size_t i=0;i<B*N;++i) dy[i]=((float)((i*17)%163)-81.0f)*0.00413f;
    for(size_t j=0;j<N;++j) db[j]=0;

    perform(variant,x,w,dy,dx,dw,db,B,K,N);
    int64_t before=clock_ns();
    for(long iteration=0;iteration<iterations;++iteration)
        perform(variant,x,w,dy,dx,dw,db,B,K,N);
    int64_t elapsed=clock_ns()-before;

    double max_error=0,checksum=0;
    for(size_t col=0;col<N;++col) {
        double reference=0;
        for(size_t row=0;row<B;++row)
            reference+=(double)dy[row*N+col];
        validate((double)db[col],reference,&max_error,&checksum,"db",col,0);
    }
    for(size_t k=0;k<K;++k)
        for(size_t col=0;col<N;++col) {
            double reference=0;
            for(size_t row=0;row<B;++row)
                reference+=(double)x[row*K+k]*(double)dy[row*N+col];
            validate((double)dw[k*N+col],reference,&max_error,&checksum,"dw",k,col);
        }
    for(size_t row=0;row<B;++row)
        for(size_t k=0;k<K;++k) {
            double reference=0;
            for(size_t col=0;col<N;++col)
                reference+=(double)dy[row*N+col]*(double)w[k*N+col];
            validate((double)dx[row*K+k],reference,&max_error,&checksum,"dx",row,k);
        }
    printf("BACKWARD_RESULT,c,%s,%zu,%zu,%zu,%ld,%" PRId64 ",%.17g,%.17g\n",
           variant,B,K,N,iterations,elapsed,max_error,checksum);
    free(x);free(w);free(dy);free(dx);free(dw);free(db);
    return 0;
}
