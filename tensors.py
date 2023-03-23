import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### Creating tensors

#scalar
scalar = torch.tensor(7)

#print(scalar.item())

vector = torch.tensor([7,7])
#print(vector.shape)
#print(vector.ndim)

MATRIX = torch.tensor([[7,8],[9,10]])

#print(MATRIX[0,0])  => 7

#print(MATRIX.shape) => dim0 =2  ,dim1 = 2

TENSOR = torch.tensor([
                        [
                            [
                                [1,2,3]
                                ],
                            [
                                [4,5,6]
                                ],
                            [
                                [7,8,9]
                                ]
                        ]
                         ])

##print(TENSOR.ndim) => 4

##print(TENSOR.shape) => 1,3,1,3


A = torch.tensor([
                  [1,2],
                  [4,5],
                  [7,8]
                  ])


##print(A.shape) => 3,2


#Create a tensor of size(3,4)

ten = torch.tensor([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12]
    ])

#print(ten.shape)  => 3,4

random_tensor = torch.rand(3,4)

#print(random_tensor)

# Create a random tensor with similear shape to an image tensor

random_image_size_tensor = torch.rand(size=(224,224,3)) #height,width,color channels(R,G,B)

#print(random_image_size_tensor.shape, random_image_size_tensor.ndim) => [224,224,3],3

