#系统
import os
import os.path as osp
from pathlib import Path
import ast
import cv2
import sys
#下载包
from vstools.utils import toml2obj
import numpy as np
import torch
#项目包
from detect_box.model import YoloOnnx
from detect_box.utils.general import increment_path, check_img_size, non_max_suppression, scale_coords
from detect_box.utils.torch_utils import select_device
from detect_box.utils.augmentations import letterbox
from detect_box.utils.plots import Annotator, colors

ROOT = osp.abspath(osp.dirname(__ROOT__))

class Vision:
    rpc_prefix = 'detect_item'

    def __init__(self, config_p):
        self.cfg = toml2obj(config_p)
        self.save_dir = self.create_savedir()
        self.model, self.stride, self.names, self.device = self.load_model()

    def infer(self, source):
        img, img0 = self.load_preimg(self.stride, source)
        result = self.predict(self.device, img, img0, self.model, source, self.save_dir, self.names)
        return result

    def create_savedir(self):
        #进行自增
        temp = ROOT + "/../" + self.cfg.output.output_path
        save_dir = increment_path(Path(temp) / self.cfg.output.name)
        save_dir.mkdir(parents=True, exist_ok=True) # parents=True 若父级目录不存在，则创建父级目录； exist_ok=True 若目录存在，则不抛出异常，忽略此操作
        return save_dir

    def load_model(self):
        
        #调用配置文件的参数
        device = select_device(self.cfg.model.device)
        weights = ROOT + "/../" + self.cfg.model.ckpt_f
        model = YoloOnnx(weights, device)
        stride, names = model.stride, model.names
        return model, stride, names, device

    def load_preimg(self, stride, source):
        imgsz = check_img_size(ast.literal_eval(self.cfg.data.imgsz), s=stride) 
        img0 = cv2.imread(source) # BGR
        #图片变为原来大小
        img = letterbox(img0, imgsz, stride=stride, auto=False)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img, img0

    def predict(self, device, img, img0, model, source, save_dir, names):
        im = torch.from_numpy(img).to(device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = model(im)
        # 非极大值抑制
        pred = non_max_suppression(pred, float(self.cfg.model.conf_thres), float(self.cfg.model.iou_thres), max_det=int(self.cfg.model.max_det))
        for i, det in enumerate(pred):
            p, img0 = source, img0.copy()
            p = Path(p)
            save_path = str(save_dir / p.name)
            annotator = Annotator(img0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round() # 坐标映射
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label =  f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
            cv2.imwrite(save_path, img0)
        # 返回结果
        pred = pred[0]
        pred = pred.cpu().numpy()
        result = {}
        result["img_path"] = save_path
        result["det_nums"] = len(pred)
        conf = []
        classes = []
        for i in range(len(pred)):
            conf.append(float(pred[i][4]))
            classes.append(self.cfg.data.names[int(pred[i][5])])
        result["conf"] = conf
        result["classes"] = classes
        return result
