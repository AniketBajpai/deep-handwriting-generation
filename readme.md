# Handwriting generation Task

Implementation of paper [Generating Sequences with Recurrent Neural Networks]((https://arxiv.org/pdf/1308.0850.pdf))

## Installation
`` conda env create -f environment.yml ``

## Unconditional Handwriting Generation

Implemented 2 models
1. LSTM with hidden dim of 900 and 1 layer
2. LSTM with hidden dim of 400 and 3 layers (skip connections)

Training:

`` python train.py --name {name} --model {model_name} [--use_gpu] [--ckpt {/path/to/ckpt}] ``

model_name: **unconditional_900** for 1 and **unconditional_400** for 2

## Conditional Handwriting Generation

Extended model 2 to generate handwriting conditioned on text input
model_name: **conditional_400**