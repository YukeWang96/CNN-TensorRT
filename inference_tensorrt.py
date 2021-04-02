#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import time
from torchvision.models import *
from utils.fp16 import network_to_half
import os
from torch2trt import torch2trt
import pandas as pd

FP32 = True
FP16 = True
INT8 = True

# make results
os.makedirs("results", exist_ok=True)

def computeTime(model, input_size=[1, 3, 224, 224], device='cuda', FP16=False):
    inputs = torch.randn(input_size)
    if device == 'cuda':
        inputs = inputs.cuda()
    if FP16:
        model = network_to_half(model)

    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(200):
        with torch.no_grad():
            _ = model(inputs)
    
    torch.cuda.synchronize() # wait for cuda to finish (cuda is asynchronous!)
    dur = time.time() - start_time
    print('Avg (ms): {:.3f}'.format(dur/200))
    return dur/200


modellist = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",  "resnext50_32x4d", \
            "resnext101_32x8d", "mnasnet1_0", "squeezenet1_0", "densenet121", "densenet169", "inception_v3"]

# resnet is enought for now
# modellist = ["resnet18", "resnet34", "resnet50"]
batchsize = 128
modellist = ["alexnet", "vgg16", "resnet18"]
results = []

for i, model_name in enumerate(modellist):
    runtimes = []

    input_size = [128, 3, 224, 224]
    mdl = globals()[model_name]
    model = mdl().cuda().eval()
    # Run raw models
    runtimes.append(computeTime(model, input_size=input_size, device="cuda", FP16=False))

    if FP32:
	    # define model
        print("FP32: {}".format(model_name), end="\t")
        mdl = globals()[model_name]
        model = mdl().cuda().eval()
        # define input
        # input_size = [1, 3, 256, 256]
        x = torch.zeros(input_size).cuda()
        # convert to tensorrt models
        model_trt = torch2trt(model, [x], max_batch_size=batchsize)
        # Run TensorRT models
        runtimes.append(computeTime(model_trt, input_size=input_size, device="cuda", FP16=False))

    if FP16:
        print("FP16: {}".format(model_name), end="\t")
        # Make FP16 tensorRT models
        mdl = globals()[model_name]
        model = mdl().eval().half().cuda()
	    # define input
        x = torch.zeros(input_size).half().cuda()
        # convert to tensorrt models
        model_trt = torch2trt(model, [x], fp16_mode=True, max_batch_size=batchsize)
        # Run TensorRT models
        runtimes.append(computeTime(model_trt, input_size=input_size, device="cuda", FP16=True))
        results.append({model_name: runtimes})

    if INT8:
        print("INT8: {}".format(model_name), end="\t")
        # Make INT8 tensorRT models
        mdl = globals()[model_name]
        model = mdl().eval().half().cuda()
        # define input
        x = torch.randn(input_size).half().cuda()
        # convert to tensorrt models
        model_trt = torch2trt(model, [x], fp16_mode=True, int8_mode=True, max_batch_size=batchsize)
        # Run TensorRT models
        runtimes.append(computeTime(model_trt, input_size=input_size, device="cuda", FP16=True))
        results.append({model_name: runtimes})

    if i == 0:
        df = pd.DataFrame({model_name: runtimes},
                         index = ["Raw", "FP32", "FP16", "INT8"])
    else:
        df[model_name] = runtimes

df.to_csv("results/xavier.csv")
df

