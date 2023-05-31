from greenthread.monkey import monkey_patch; monkey_patch()
from greenthread.green import *
import os.path as osp
import os

from vstools.utils import toml2obj

from mqsrv.client import make_client


def main():
    cfg_obj = toml2obj("config/config.toml")
    client = make_client()
    caller = client.get_caller(cfg_obj.mqsrv.rpc_queue)
    
    img_name = input("请输入要检测的图片路径:")
    img_path = os.path.join(cfg_obj.data.data_path, img_name)
    print(img_path)
    
    exc, result = caller.detect_item_infer(img_path)
    print(result)
    
    # 释放客户端
    client.release()


if __name__ == '__main__':
    main('pyamqp://')
