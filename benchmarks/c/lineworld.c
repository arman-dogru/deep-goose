// Exact dynamics, exploration schedule and Q update as examples/rl/lineworld.goose.
#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int64_t now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)) abort();
    return (int64_t)t.tv_sec*1000000000LL+t.tv_nsec;
}
int main(void) {
    enum { EPISODES=3000, MAX_STEPS=64 };
    float q[16]={0};
    int64_t seed=12345, transitions=0, completed=0;
    int64_t before=now_ns();
    for (int ep=0;ep<EPISODES;++ep) {
        int state=0;
        for (int step=0;step<MAX_STEPS;++step) {
            seed=(seed*1664525+1013904223)%4294967296LL;
            int explore=(seed%1000)<100;
            int action= explore ? ((seed%2)==0 ? 0 : 1)
                                : (q[state*2+1]>=q[state*2] ? 1 : 0);
            int next= action==0 ? (state>0 ? state-1 : 0)
                                : (state<7 ? state+1 : 7);
            int terminal=next==7;
            int index=state*2+action;
            float target=terminal?1.0f:0.0f;
            if (!terminal) {
                float best=q[next*2] > q[next*2+1] ? q[next*2] : q[next*2+1];
                target+=0.95f*best;
            }
            q[index]+=0.2f*(target-q[index]);
            transitions++;
            if(terminal) { completed++; break; }
            state=next;
        }
    }
    int64_t elapsed=now_ns()-before;
    int state=0, eval_steps=0;
    for (int i=0;i<MAX_STEPS;++i) {
        int action=q[state*2+1]>=q[state*2] ? 1 : 0;
        state=action==0?(state>0?state-1:0):(state<7?state+1:7);
        eval_steps++;
        if(state==7)break;
    }
    if(state!=7 || eval_steps!=7 || completed!=EPISODES)return 3;
    printf("RL_RESULT,c,%d,%" PRId64 ",%" PRId64 ",%.9g,%.9g,%d\n",
           EPISODES,elapsed,transitions,q[1],q[13],eval_steps);
    return 0;
}
