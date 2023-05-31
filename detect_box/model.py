from pathlib import Path
import json

import torch.nn as nn
import torch
import numpy as np


class YoloOnnx(nn.Module):
    def __init__(self, ckpt, device, data=None):
        super().__init__()
        onnx, jit = self._model_type(ckpt)
        stride = 32  # default stride

        if onnx:
            cuda = torch.cuda.is_available and device.type != 'cpu'
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(ckpt, providers=providers)
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
        elif jit:
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(ckpt, _extra_files=extra_files)
            model.float()
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        
        self.__dict__.update(locals()) # 将所有局部变量赋值给当前对象的属性

    def forward(self, im):
        # b, ch, h, w = im.shape  # batch, channel, height, width
        # im = im.half() # 半精度浮点数表示 节省空间提高效率 但降低精度
        if self.jit:
            y = self.model(im)[0]
        elif self.onnx:
            im = im.cpu().numpy()
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return y
    
    @staticmethod
    def _model_type(p):
        # 依照model path返回model type， i.e. path='path/to/model.onnx' -> type=onnx
        # suffixes = [".pt",".onnx",".torchscript"] + [".xml"]
        suffixes = [".pt",".onnx",".torchscript"]
        p = Path(p).name # 从路径得到文件名， i.e. path='path/to/model.onnx' -> model.onnx
        pt, onnx, jit = (s in p for s in suffixes)
        return onnx, jit