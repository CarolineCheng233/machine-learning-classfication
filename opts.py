# from .config import *
from argparse import ArgumentParser
from mmcv import Config


def parse():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'])
    parser.add_argument('--file_name', type=str, required=False)
    parser.add_argument('--bert_path', type=str, required=False)
    args = Config.fromfile("config.py")
    args.merge_from_dict(parser.parse_args())
    # if "file_name" in arguments:
    #     args["file_name"] = arguments.file_name
    # else:
    #     args["file_name"] = file_name
    # if "bert_path" in arguments:
    #     args["bert_path"] = arguments.bert_path
    # else:
    #     args["bert_path"] = bert_path

    return args
