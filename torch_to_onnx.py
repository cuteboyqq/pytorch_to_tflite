import torch
from torchvision.models import mobilenet_v2
import torch.onnx
import os
import torch.nn as nn
import torch.nn.functional as F
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


#img_size = (32, 32)
#batch_size = 1
onnx_model_path = 'resnet.onnx'
input_size = 32
path = "/home/ali/repVGG/model/2022-07-03/resnet-Size32-16-32-64-64-b2-2-3-2.pt"
model = torch.load(path, map_location=torch.device('cpu')) 
#model = mobilenet_v2()
model.eval()

#sample_input = torch.rand((batch_size, 3, *img_size))
sample_input = torch.randn(1, 3, input_size, input_size, device = 'cpu')

#y = model(sample_input)

torch.onnx.export(
    model,
    sample_input, 
    onnx_model_path,
    export_params=True,  # store the trained parameter weights inside the model file
    verbose=True,
    input_names=['input'],
    output_names=['output'],
    opset_version=9
)