import os
import platform
import sys
from pathlib import Path
from loguru import logger
import os.path as osp
os.environ['GREEN_BACKEND'] = 'gevent'

import eventlet
eventlet.monkey_patch()
from mqsrv.server import make_server, run_server
from vstools.utils import toml2obj

from detect_box.vision import Vision


# 对模型进行运用
def run():
    cfg_obj = toml2obj("config/config.toml")
    rpc_queue = cfg_obj.mqsrv.rpc_queue
    server = make_server(rpc_routing_key=rpc_queue)
    # 调用模型的参数
    vision = Vision("config/config.toml")
    #注册模型的推断功能
    server.register_rpc(vision.infer)
    run_server(server)

#开始运行模型
def run_wrapper():
    try:
        run()
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    run_wrapper()