# Vessel_Trajectory_Prediction

Unofficial Implementation of Paper: __[Deep Learning Methods for Vessel Trajectory
Prediction based on Recurrent Neural Networks] (https://ieeexplore.ieee.org/document/9492102)__

![vesselTracking](https://github.com/user-attachments/assets/8dd6ef95-6084-4a32-8e5a-27651a37f904)

## Steps to use this repo:

### import necessary modules
> __import torch__ \
> __from vTrack.vTrack import vTrack__

### Declare hyper-parameters
> __input_size = 4__ \
> __output_size = 2__ \
> __hidden_size = 16__ \
> __num_layers = 3__ \
> __seq_length = (12, 6)__ _# 12 input sequences, 6 output sequences_ \
> __batch_size = 8__

### input sample 
> __x = torch.rand(batch_size, seq_length[0], input_size)__ _# (8, 12, 4)_

### create object of the Model
> __model = vTrack(input_size, hidden_size, num_layers, output_size, seq_length[1])__

### Feed the sample to the Model
> __out = model(x)__ _# (batch, out_seq_length, output_size) : (8, 6, 2)_

> Errors and Omissions expected!
