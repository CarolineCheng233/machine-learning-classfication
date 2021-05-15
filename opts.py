# from .config import *
from argparse import ArgumentParser
from mmcv import Config, DictAction


def parse():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--cfg', type=str, default='config.py')
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default={})
    args = parser.parse_args()
    cfgs = Config.fromfile("config.py")
    cfgs.merge_from_dict(args.cfg_options)
    cfgs.mode = args.mode

    return cfgs
