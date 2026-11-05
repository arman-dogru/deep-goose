#!/usr/bin/env python3
"""Same 256-example, 2->1 linear regression as Goose/C Adam implementations.

Computes manual dense gradients and bias-corrected Adam; excludes data
construction and final loss evaluation from timed training region.
Usage: python regression_adam.py <python_loop|numpy|torch_cpu>
Output: ADAM_RESULT,implementation,epochs,elapsed_ns,loss,w0,w1,b
"""
from __future__ import annotations
import math
import sys
import time

EPOCHS, BATCH, LR = 250, 256, 0.03
BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-8


def train(implementation):
    data = [(((i % 17)-8)/8.0, (((i*7)%19)-9)/9.0) for i in range(BATCH)]
    labels = [1.5*x0 - 2.0*x1 + 0.25 for x0,x1 in data]
    beta1_power, beta2_power = 1.0, 1.0

    if implementation == "python_loop":
        w = [0.0, 0.0]
        b = [0.0]
        mw, vw, mb, vb = [0.0,0.0], [0.0,0.0], [0.0], [0.0]
        predicted, dy = [0.0]*BATCH, [0.0]*BATCH
        dx = [0.0]*(BATCH*2)
        def epoch(c1,c2):
            for i,(f0,f1) in enumerate(data):
                predicted[i] = b[0]+f0*w[0]+f1*w[1]
            for i in range(BATCH):
                dy[i] = (2.0/BATCH)*(predicted[i]-labels[i])
            gb = sum(dy)
            gw = [sum(data[i][k]*dy[i] for i in range(BATCH)) for k in range(2)]
            for i in range(BATCH):
                dx[2*i] = dy[i]*w[0]
                dx[2*i+1] = dy[i]*w[1]
            for k in range(2):
                g=gw[k]
                mw[k]=BETA1*mw[k]+(1-BETA1)*g
                vw[k]=BETA2*vw[k]+(1-BETA2)*g*g
                w[k] -= LR*(mw[k]/c1)/(math.sqrt(vw[k]/c2)+EPSILON)
            mb[0]=BETA1*mb[0]+(1-BETA1)*gb
            vb[0]=BETA2*vb[0]+(1-BETA2)*gb*gb
            b[0] -= LR*(mb[0]/c1)/(math.sqrt(vb[0]/c2)+EPSILON)
        def parameters():
            return w[0],w[1],b[0]
    elif implementation == "numpy":
        import numpy as np
        x=np.asarray(data,dtype=np.float32)
        target=np.asarray(labels,dtype=np.float32).reshape(BATCH,1)
        w=np.zeros((2,1),dtype=np.float32)
        b=np.zeros((1,),dtype=np.float32)
        mw=np.zeros_like(w);vw=np.zeros_like(w)
        mb=np.zeros_like(b);vb=np.zeros_like(b)
        def epoch(c1,c2):
            predicted=x@w+b
            dy=np.float32(2.0/BATCH)*(predicted-target)
            gw=x.T@dy
            gb=dy.sum(axis=0)
            dx=dy@w.T
            np.multiply(mw,BETA1,out=mw); mw[:]+=np.float32(1-BETA1)*gw
            np.multiply(vw,BETA2,out=vw); vw[:]+=np.float32(1-BETA2)*gw*gw
            np.multiply(mb,BETA1,out=mb); mb[:]+=np.float32(1-BETA1)*gb
            np.multiply(vb,BETA2,out=vb); vb[:]+=np.float32(1-BETA2)*gb*gb
            w[:]-=LR*(mw/c1)/(np.sqrt(vw/c2)+EPSILON)
            b[:]-=LR*(mb/c1)/(np.sqrt(vb/c2)+EPSILON)
            return dx
        def parameters():
            return float(w[0,0]),float(w[1,0]),float(b[0])
    elif implementation == "torch_cpu":
        import torch
        torch.set_num_threads(1)
        x=torch.tensor(data,dtype=torch.float32)
        target=torch.tensor(labels,dtype=torch.float32).reshape(BATCH,1)
        w=torch.zeros((2,1),dtype=torch.float32)
        b=torch.zeros((1,),dtype=torch.float32)
        mw=torch.zeros_like(w);vw=torch.zeros_like(w)
        mb=torch.zeros_like(b);vb=torch.zeros_like(b)
        def epoch(c1,c2):
            predicted=x@w+b
            dy=(2.0/BATCH)*(predicted-target)
            gw=x.T@dy
            gb=dy.sum(dim=0)
            dx=dy@w.T
            mw.mul_(BETA1).add_((1-BETA1)*gw)
            vw.mul_(BETA2).add_((1-BETA2)*gw*gw)
            mb.mul_(BETA1).add_((1-BETA1)*gb)
            vb.mul_(BETA2).add_((1-BETA2)*gb*gb)
            w.sub_(LR*(mw/c1)/(torch.sqrt(vw/c2)+EPSILON))
            b.sub_(LR*(mb/c1)/(torch.sqrt(vb/c2)+EPSILON))
            return dx
        def parameters():
            return float(w[0,0]),float(w[1,0]),float(b[0])
    else:
        raise ValueError(implementation)

    start=time.perf_counter_ns()
    for _ in range(EPOCHS):
        beta1_power*=BETA1
        beta2_power*=BETA2
        epoch(1-beta1_power,1-beta2_power)
    elapsed=time.perf_counter_ns()-start
    w0,w1,bias=parameters()
    loss=sum((w0*x0+w1*x1+bias-y)**2 for (x0,x1),y in zip(data,labels))/BATCH
    if loss>=1e-4 or not math.isfinite(loss):
        raise AssertionError(f"Adam failed to converge: loss={loss}")
    print(f"ADAM_RESULT,{implementation},{EPOCHS},{elapsed},{loss:.9g},{w0:.9g},{w1:.9g},{bias:.9g}")


if __name__ == "__main__":
    if len(sys.argv)!=2:
        raise SystemExit("usage: regression_adam.py <python_loop|numpy|torch_cpu>")
    train(sys.argv[1])
