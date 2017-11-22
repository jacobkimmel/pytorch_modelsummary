# PyTorch Model Summarizer

This tool summarizes a [PyTorch](https://pytorch.org) model, with behavior similar to the [Keras](https://keras.io) `model.summary()` method.  

## Usage

To use the summarizer, simply import the `ModelSummary` class, then provide a model and an input size for estimation.

```python
# Define a model
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()

        self.conv0 = nn.Conv2d(1, 16, kernel_size=3, padding=5)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3)

    def forward(self, x):
        h = self.conv0(x)
        h = self.conv1(h)
        return h

model = Model()

# Summarize Model
from pytorch_modelsummary import ModelSummary

ms = ModelSummary(model, input_size=(1, 1, 256, 256))

# Prints
# ------
# Name    Type               InSz              OutSz  Params
# 0  conv0  Conv2d   [1, 1, 256, 256]  [1, 16, 264, 264]     160
# 1  conv1  Conv2d  [1, 16, 264, 264]  [1, 32, 262, 262]    4640

# ms.summary is a Pandas DataFrame
print(ms.summary['Params'])
# 0     160
# 1    4640
# Name: Params, dtype: int64
```

## Development

This tool is a product of the [Laboratory of Cell Geometry](https://cellgeometry.ucsf.edu/) at the [University of California, San Francisco](https://ucsf.edu).
