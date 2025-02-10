# Vessel_Trajectory_Prediction

Unofficial Implementation of Paper: Deep Learning Methods for Vessel Trajectory
Prediction based on Recurrent Neural Networks

![vesselTracking](https://github.com/user-attachments/assets/8dd6ef95-6084-4a32-8e5a-27651a37f904)

## Steps to use this repo:

### import necessary modules
import torch \
from vTrack.vTrack import vTrack

### Declare hyper-parameters
input_size = 4 \
output_size = 2 \
hidden_size = 16 \
num_layers = 3 \
seq_length = (12, 6) _# 12 input sequences, 6 output sequences_ \
batch_size = 8

### input sample 
x = torch.rand(batch_size, seq_length[0], input_size)

### create object of Model
model = vTrack(input_size, hidden_size, num_layers, output_size, seq_length[1])

### Feed the sample to the Model
out = model(x) _# (batch, out_seq_length, output_size) : (8, 6, 2)_
