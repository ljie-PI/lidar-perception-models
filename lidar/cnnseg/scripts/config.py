#!/usr/bin/env python

import json
import traceback

class ModelConfig(object):

    def __init__(self, config_file):
        try:
            js_config = json.load(open(config_file))
        except Exception as e:
            raise Exception("Parsing config failed: " + traceback.format_exc(e))
        # config for input
        self.in_channel = js_config.get("in_channel", 6)
        self.dis_chan_idx = js_config.get("dis_chan_idx", -1)
        self.mask_chan_idx = js_config.get("mask_chan_idx", 5)
        self.out_channel = js_config.get("out_channel", 12)
        self.height = js_config.get("height", 672)
        self.width = js_config.get("width", 672)
        self.batch_size = js_config.get("batch_size", 128)
        self.thread_num = js_config.get("thread_num", 8)

        # config for losses
        self.pos_weight = js_config.get("pos_weight", 10)
        self.fl_gamma = js_config.get("fl_gamma", 2)
        self.fl_alpha = js_config.get("fl_alpha", 0.25)
        self.unmask_ratio = js_config.get("unmask_ratio", 0.02)
        self.category_label_thr = js_config.get("category_label_thr", 0.4)
        self.confidence_label_thr = js_config.get("confidence_label_thr", 0.1)
        self.merge_loss = js_config.get("merge_loss", True)
        self.loss_weights = js_config.get("loss_weights", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.dis_crit_range = js_config.get("dis_crit_range", 30)
        self.dis_smooth_factor = js_config.get("dis_smooth_factor", 0.5)
        self.dis_loss_decay = js_config.get("dis_loss_decay", 0.6)
        self.category_loss = "focal_loss"
        self.confidence_loss = "focal_loss"

        # config for training process
        self.learning_rate = js_config.get("learning_rate", 5e-5)
        self.max_steps = js_config.get("max_steps", 500000)
        self.save_ckpts_steps = js_config.get("save_ckpts_steps", 5000)
        self.enable_summary = js_config.get("enable_summary", False)
        self.save_summary_steps = js_config.get("save_summary_steps", 500)
        self.max_keep_ckpts = js_config.get("max_keep_ckpts", 3)