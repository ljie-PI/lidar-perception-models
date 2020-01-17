#!/usr/bin/env python

import os
import json
import logging
import torch


def latest_checkpoint(model_dir, model_name):
    ckpt_info_path = os.path.join(model_dir, "{}-checkpoints.json".format(model_name))
    logging.info("ckpt_info_path: {}".format(ckpt_info_path))
    if not os.path.exists(ckpt_info_path):
        logging.info("{} doesn't exist".format(ckpt_info_path))
        return None, 0
    with open(ckpt_info_path, "r") as f:
        ckpt_dict = json.loads(f.read())
    latest_ckpt = ckpt_dict["latest_ckpt"]
    latest_step = ckpt_dict["latest_step"]
    ckpt_file_name = os.path.join(model_dir, latest_ckpt)
    if not os.path.exists(ckpt_file_name):
        logging.info("{} doesn't exist".format(ckpt_file_name))
        return None, 0
    return ckpt_file_name, latest_step


def save_model(model_dir,
               model,
               model_name,
               global_step,
               max_to_keep=5):
    ckpt_info_path = os.path.join(model_dir, "{}-checkpoints.json".format(model_name))
    if not os.path.exists(ckpt_info_path):
        ckpt_info_dict = {"latest_ckpt": {}, "latest_step": 0, "all_ckpts": []}
    else:
        with open(ckpt_info_path, "r") as f:
            ckpt_info_dict = json.loads(f.read())
    ckpt_filename = "{}-step-{}.ckpt".format(model_name, global_step)
    ckpt_path = os.path.join(model_dir, ckpt_filename)
    torch.save(model.state_dict(), ckpt_path)

    ckpt_info_dict["latest_ckpt"] = ckpt_filename
    ckpt_info_dict["latest_step"] = global_step
    ckpts_to_delete = []
    saved_model_num = len(ckpt_info_dict["all_ckpts"])
    if saved_model_num >= max_to_keep:
        for _ in range(saved_model_num - max_to_keep + 1):
            ckpts_to_delete.append(
                os.path.join(model_dir, ckpt_info_dict["all_ckpts"].pop(0)))
    ckpt_info_dict["all_ckpts"].append(ckpt_filename)
    with open(ckpt_info_path, "w") as f:
        f.write(json.dumps(ckpt_info_dict, indent=2))
    for ckpt_to_delete in ckpts_to_delete:
        if os.path.exists(ckpt_to_delete):
            os.remove(ckpt_to_delete)


def restore_model(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise ValueError("checkpoint {} not exist.".format(ckpt_path))
    model.load_state_dict(torch.load(ckpt_path))
