#!/usr/bin/env python3
"""Matched regression workload: 256x2 batch, one dense layer, SGD for 200 epochs.

Usage: python regression_train.py <python_loop|numpy|torch_cpu>
Output: TRAIN_RESULT,implementation,epochs,elapsed_ns,loss,w0,w1,b
Dataset preparation and final loss evaluation are excluded from timing.
"""
from __future__ import annotations
import sys
import time


def train(implementation):
    B, epochs, lr = 256, 200, 0.1
    data = [(((i % 17) - 8) / 8.0, (((i * 7) % 19) - 9) / 9.0) for i in range(B)]
    labels = [1.5*x0-2.0*x1+0.25 for x0,x1 in data]
    if implementation == "python_loop":
        w = [0.0, 0.0]
        b = [0.0]
        prediction = [0.0]*B
        grad_y = [0.0]*B
        grad_x = [0.0]*(B*2)
        def epoch():
            for i,(x0,x1) in enumerate(data):
                prediction[i] = b[0] + x0*w[0] + x1*w[1]
            for i in range(B):
                grad_y[i] = 2.0/B*(prediction[i]-labels[i])
            gb = sum(grad_y)
            gw0 = sum(data[i][0]*grad_y[i] for i in range(B))
            gw1 = sum(data[i][1]*grad_y[i] for i in range(B))
            for i in range(B):
                grad_x[i*2] = grad_y[i]*w[0]
                grad_x[i*2+1] = grad_y[i]*w[1]
            w[0] -= lr*gw0
            w[1] -= lr*gw1
            b[0] -= lr*gb
        def parameters():
            return float(w[0]), float(w[1]), float(b[0])
    elif implementation == "numpy":
        import numpy as np
        x = np.asarray(data, dtype=np.float32)
        target = np.asarray(labels, dtype=np.float32).reshape(B, 1)
        w = np.zeros((2,1), dtype=np.float32)
        b = np.zeros((1,), dtype=np.float32)
        def epoch():
            prediction = x @ w + b
            grad_y = 2.0/B * (prediction-target)
            grad_w = x.T @ grad_y
            grad_b = grad_y.sum(axis=0)
            grad_x = grad_y @ w.T  # full dense backward path
            w[:] -= lr*grad_w
            b[:] -= lr*grad_b
            return grad_x
        def parameters():
            return float(w[0,0]), float(w[1,0]), float(b[0])
    elif implementation == "torch_cpu":
        import torch
        torch.set_num_threads(1)
        x = torch.tensor(data, dtype=torch.float32)
        target = torch.tensor(labels, dtype=torch.float32).reshape(B,1)
        w = torch.zeros((2,1), dtype=torch.float32)
        b = torch.zeros((1,), dtype=torch.float32)
        def epoch():
            prediction = x @ w + b
            grad_y = (2.0/B)*(prediction-target)
            grad_w = x.T @ grad_y
            grad_b = grad_y.sum(dim=0)
            grad_x = grad_y @ w.T
            w.sub_(lr*grad_w)
            b.sub_(lr*grad_b)
            return grad_x
        def parameters():
            return float(w[0,0]), float(w[1,0]), float(b[0])
    else:
        raise SystemExit(f"unknown implementation {implementation}")

    start = time.perf_counter_ns()
    for _ in range(epochs):
        epoch()
    elapsed_ns = time.perf_counter_ns() - start
    w0,w1,bias = parameters()
    loss = sum(((w0*x0 + w1*x1 + bias)-y)**2 for (x0,x1),y in zip(data,labels))/B
    if loss >= 1e-4:
        raise AssertionError(f"did not converge: {loss}")
    print(f"TRAIN_RESULT,{implementation},{epochs},{elapsed_ns},{loss:.9g},{w0:.9g},{w1:.9g},{bias:.9g}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: regression_train.py <python_loop|numpy|torch_cpu>")
    train(sys.argv[1])
