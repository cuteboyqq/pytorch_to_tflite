import os
import torch
import argparse

from models.models import *

class ONNXExportableModel(torch.nn.Module):
    def __init__(self, model, SSD=False):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        # detections = [(det['boxes'], det['labels'], det['scores']) for det in detections]
        return self.model(*args, **kwargs)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    #parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--cfg', type=str, required=True, help='*.cfg path')
    opt = parser.parse_args()
    print(opt)
    model = Darknet(opt.cfg, opt.img_size)
    # print(opt.weights)
    model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model'])
  
    dummy_input = torch.randn(1, 3, opt.img_size, opt.img_size, device='cpu')
    filename = os.path.splitext(opt.weights)[0]
    print(filename)
    torch.onnx.export(ONNXExportableModel(model), dummy_input, f'{filename}.onnx', opset_version=9, verbose=False,
                  input_names = ['data'],   # the model's input names
                  output_names = ['yolo1','yolo2','yolo3'], # the model's output names
                  )
