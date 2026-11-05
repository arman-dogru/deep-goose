#!/usr/bin/env python3
"""Pure Python control implementation for deterministic tabular LineWorld Q-learning."""
import time

EPISODES=3000
MAX_STEPS=64
seed=12345
transitions=0
completed=0
q=[0.0]*16
start=time.perf_counter_ns()
for episode in range(EPISODES):
    state=0
    for step in range(MAX_STEPS):
        seed=(seed*1664525+1013904223)%4294967296
        explore=(seed%1000)<100
        action=(0 if seed%2 == 0 else 1) if explore else (1 if q[state*2+1]>=q[state*2] else 0)
        next_state=max(0,state-1) if action==0 else min(7,state+1)
        terminal=next_state==7
        index=state*2+action
        target=1.0 if terminal else 0.95*max(q[next_state*2],q[next_state*2+1])
        q[index]+=0.2*(target-q[index])
        transitions+=1
        if terminal:
            completed+=1
            break
        state=next_state
elapsed_ns=time.perf_counter_ns()-start
state=0
eval_steps=0
for step in range(MAX_STEPS):
    action=1 if q[state*2+1]>=q[state*2] else 0
    state=max(0,state-1) if action==0 else min(7,state+1)
    eval_steps+=1
    if state==7:
        break
assert state==7 and eval_steps==7 and completed==EPISODES
print(f"RL_RESULT,python_loop,{EPISODES},{elapsed_ns},{transitions},{q[1]:.9g},{q[13]:.9g},{eval_steps}")
