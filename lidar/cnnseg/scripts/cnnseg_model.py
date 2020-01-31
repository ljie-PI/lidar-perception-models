#!/usr/bin/env python

import tensorflow as tf
from common_util import CommonUtil

class CNNSegModel(object):
    VARIABLE_SCOPE = "cnn_seg"

    def __init__(self, config, inputs, labels, mask, example_ids, is_training):
        self._config = config
        self._is_training = is_training
        with tf.variable_scope(self.VARIABLE_SCOPE):
            self._build_graph(inputs, mask)
            if is_training:
                self._compute_loss(inputs, labels, mask, example_ids)
            elif labels is not None:
                self._compute_metric(labels, mask)

    def _conv_layer(self,
                    inputs,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding="SAME",
                    use_activation=True,
                    use_batch_norm=True,
                    scope=None):
        with tf.variable_scope(scope):
            filter_shape = [kernel_size[0], kernel_size[1], in_channels, out_channels]
            filter_weight = tf.get_variable(
                "filter_weight",
                shape=filter_shape,
                # initializer=tf.glorot_uniform_initializer()
                initializer=tf.initializers.he_uniform()
            )
            bias = tf.get_variable(
                "bias",
                shape=[out_channels],
                initializer=tf.zeros_initializer()
            )
            conv = tf.nn.conv2d(
                inputs,
                filter=filter_weight,
                strides=stride,
                padding=padding
            )
            bias_add = tf.nn.bias_add(conv, bias)

            if use_batch_norm:
                batch_norm = tf.layers.batch_normalization(
                    bias_add,
                    training=self._is_training
                )
            else:
                batch_norm = bias_add

            if use_activation:
                return tf.nn.relu(batch_norm)
            else:
                return batch_norm
    
    def _deconv_layer(self,
                      inputs,
                      in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding="SAME",
                      out_shape=None,
                      use_activation=True,
                      use_batch_norm=True,
                      scope=None):
        with tf.variable_scope(scope):
            filter_shape = [kernel_size[0], kernel_size[1], out_channels, in_channels]
            filter_weight = tf.get_variable(
                "filter_weight",
                shape=filter_shape,
                # initializer=tf.glorot_uniform_initializer()
                initializer=tf.initializers.he_uniform()
            )
            bias = tf.get_variable(
                "bias",
                shape=[out_channels],
                initializer=tf.zeros_initializer()
            )
            deconv = tf.nn.conv2d_transpose(
                inputs,
                filter=filter_weight,
                output_shape=out_shape,
                strides=stride,
                padding=padding
            )
            bias_add = tf.nn.bias_add(deconv, bias)
            
            if use_batch_norm:
                batch_norm = tf.layers.batch_normalization(
                    bias_add,
                    training=self._is_training
                )
            else:
                batch_norm = bias_add

            if use_activation:
                return tf.nn.relu(batch_norm)
            else:
                return batch_norm
    
    def _mask_and_flatten(self, tensor, mask):
        return tf.reshape(
            tf.multiply(tensor, mask),
            [-1, 1]
        )

    def _sigmoid_cross_entropy_loss(self, logits, labels, ext_mask):
        flatten_logits = self._mask_and_flatten(logits, ext_mask)
        flatten_labels = self._mask_and_flatten(labels, ext_mask)
        flatten_labels = tf.to_float(tf.greater_equal(flatten_labels, 0.5))
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=flatten_labels,
            logits=flatten_logits
        )
        return tf.reduce_sum(losses) / tf.reduce_sum(ext_mask) / self._config.batch_size

    def _soft_target_cross_entropy_loss(self, logits, labels, ext_mask):
        flatten_logits = self._mask_and_flatten(logits, ext_mask)
        flatten_labels = self._mask_and_flatten(labels, ext_mask)
        losses = tf.nn.weighted_cross_entropy_with_logits(
            labels=flatten_labels,
            logits=flatten_logits,
            pos_weight=self._config.pos_weight
        )
        return tf.reduce_sum(losses) / tf.reduce_sum(ext_mask) / self._config.batch_size

    def _weighted_cross_entropy_loss(self, logits, labels, ext_mask):
        flatten_logits = self._mask_and_flatten(logits, ext_mask)
        flatten_labels = self._mask_and_flatten(labels, ext_mask)
        flatten_labels = tf.to_float(tf.greater_equal(flatten_labels, 0.5))
        losses = tf.nn.weighted_cross_entropy_with_logits(
            labels=flatten_labels,
            logits=flatten_logits,
            pos_weight=self._config.pos_weight
        )
        return tf.reduce_sum(losses) / tf.reduce_sum(ext_mask) / self._config.batch_size

    def _focal_loss(self, preds, labels, ext_mask):
        flatten_preds = self._mask_and_flatten(preds, ext_mask)
        flatten_labels = self._mask_and_flatten(labels, ext_mask)
        flatten_labels = tf.to_float(tf.greater_equal(flatten_labels, 0.5))
        gamma = self._config.fl_gamma
        alpha = self._config.fl_alpha
        epsilon = 1e-8
        losses = -1 * flatten_labels * tf.pow(1-flatten_preds, gamma) * tf.log(flatten_preds+epsilon) * alpha \
                 - (1-flatten_labels) * tf.pow(flatten_preds, gamma) * tf.log(1-flatten_preds+epsilon)
        return tf.reduce_sum(losses * 100) / tf.reduce_sum(ext_mask) / self._config.batch_size

    def _distance_aware_focal_loss(self, preds, labels, distance, ext_mask):
        flatten_preds = self._mask_and_flatten(preds, ext_mask)
        flatten_labels = self._mask_and_flatten(labels, ext_mask)
        flatten_labels = tf.to_float(tf.greater_equal(flatten_labels, 0.5))
        gamma = self._config.fl_gamma
        pos_weight = self._config.pos_weight
        epsilon = 1e-8
        losses = -1 * flatten_labels * tf.pow(1-flatten_preds, gamma) * tf.log(flatten_preds+epsilon) * pos_weight \
                 - (1-flatten_labels) * tf.pow(flatten_preds, gamma) * tf.log(1-flatten_preds+epsilon)
        dis_crit_range = self._config.dis_crit_range
        dis_smooth_factor = self._config.dis_smooth_factor
        dis_loss_decay = self._config.dis_loss_decay
        flatten_distance = tf.reshape(distance, [-1, 1])
        distance_factor = 1 / (1 + tf.exp(dis_smooth_factor * (flatten_distance - dis_crit_range))) * (1 - dis_loss_decay) + dis_loss_decay
        losses = losses * distance_factor
        return tf.reduce_sum(losses * 100) / tf.reduce_sum(ext_mask) / self._config.batch_size

    def _softmax_cross_entropy_loss(self, logits, labels, mask):
        flatten_logits = tf.reshape(tf.multiply(logits, mask), [-1, 5])
        flatten_labels = tf.reshape(tf.multiply(labels, mask), [-1, 5])
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=flatten_labels,
            logits=flatten_logits
        )
        return tf.reduce_sum(losses) / tf.reduce_sum(mask) / self._config.batch_size

    def _mean_squared_error(self, preds, labels, mask):
        mask_preds = tf.multiply(preds, mask)
        flatten_preds = tf.reshape(mask_preds, [-1, 1])
        mask_labels = tf.multiply(labels, mask)
        flatten_labels = tf.reshape(mask_labels, [-1, 1])
        mse_sum = tf.losses.mean_squared_error(
            flatten_labels,
            flatten_preds,
            reduction=tf.losses.Reduction.SUM
        )
        return mse_sum / tf.reduce_sum(mask) / self._config.batch_size

    def _cat_label_masked_mean_squared_error(self, preds, labels, mask, cat_labels):
        flatten_preds = self._mask_and_flatten(preds, mask)
        cat_lbl_mask = tf.to_float(tf.greater_equal(cat_labels, 0.5))
        flatten_labels = self._mask_and_flatten(labels, cat_lbl_mask)
        mse_sum = tf.losses.mean_squared_error(
            flatten_labels,
            flatten_preds,
            reduction=tf.losses.Reduction.SUM
        )
        return mse_sum / tf.reduce_sum(mask) / self._config.batch_size

    def _rand_ext_masked_mean_squared_error(self, preds, labels, ext_mask, cat_labels):
        flatten_preds = self._mask_and_flatten(preds, ext_mask)
        cat_lbl_mask = tf.to_float(tf.greater_equal(cat_labels, 0.5))
        flatten_labels = self._mask_and_flatten(labels, cat_lbl_mask)
        mse_sum = tf.losses.mean_squared_error(
            flatten_labels,
            flatten_preds,
            reduction=tf.losses.Reduction.SUM
        )
        return mse_sum / tf.reduce_sum(ext_mask) / self._config.batch_size

    def _mean_iou_metric(self, preds, labels, num_classes):
        labels = tf.math.argmax(labels, axis=-1)    
        preds = tf.math.argmax(preds, axis=-1)
        return tf.metrics.mean_iou(labels, preds, num_classes)
    
    def _root_mean_squared_error_metric(self, scores, labels, mask):
        mask_scores = tf.multiply(scores, mask)
        return tf.metrics.root_mean_squared_error(labels, mask_scores)

    def _build_graph(self, inputs, mask):
        tf.logging.info("shape of inputs is {}".format(inputs.shape))

        conv0_1 = self._conv_layer(inputs, self._config.in_channel, 24, (1, 1), 1, padding="VALID", scope="conv0_1")
        conv0 = self._conv_layer(conv0_1, 24, 24, (3, 3), 1, scope="conv0")
        tf.logging.info("shape of conv0 is {}".format(conv0.shape))

        conv1_1 = self._conv_layer(conv0, 24, 48, (3, 3), 2, scope="conv1_1")
        conv1 = self._conv_layer(conv1_1, 48, 48, (3, 3), 1, scope="conv1")
        tf.logging.info("shape of conv1 is {}".format(conv1.shape))

        conv2_1 = self._conv_layer(conv1, 48, 64, (3, 3), 2, scope="conv2_1")
        conv2_2 = self._conv_layer(conv2_1, 64, 64, (3, 3), 1, scope="conv2_2")
        conv2 = self._conv_layer(conv2_2, 64, 64, (3, 3), 1, scope="conv2")
        tf.logging.info("shape of conv2 is {}".format(conv2.shape))

        conv3_1 = self._conv_layer(conv2, 64, 96, (3, 3), 2, scope="conv3_1")
        conv3_2 = self._conv_layer(conv3_1, 96, 96, (3, 3), 1, scope="conv3_2")
        conv3 = self._conv_layer(conv3_2, 96, 96, (3, 3), 1, scope="conv3")
        tf.logging.info("shape of conv3 is {}".format(conv3.shape))

        conv4_1 = self._conv_layer(conv3, 96, 128, (3, 3), 2, scope="conv4_1")
        conv4_2 = self._conv_layer(conv4_1, 128, 128, (3, 3), 1, scope="conv4_2")
        conv4 = self._conv_layer(conv4_2, 128, 128, (3, 3), 1, scope="conv4")
        tf.logging.info("shape of conv4 is {}".format(conv4.shape))

        conv5_1 = self._conv_layer(conv4, 128, 192, (3, 3), 2, scope="conv5_1")
        conv5 = self._conv_layer(conv5_1, 192, 192, (3, 3), 1, scope="conv5")
        tf.logging.info("shape of conv5 is {}".format(conv5.shape))

        deconv5_1 = self._conv_layer(conv5, 192, 192, (3, 3), 1, scope="deconv5_1")
        tf.logging.info("shape of deconv5_1 is {}".format(deconv5_1.shape))

        deconv4 = self._deconv_layer(deconv5_1, 192, 128, (4, 4), 2,
                                     out_shape=tf.shape(conv4), scope="deconv4")
        concat4 = tf.concat([conv4, deconv4], axis=3)
        deconv4_1 = self._conv_layer(concat4, 256, 128, (3, 3), 1, scope="deconv4_1")
        tf.logging.info("shape of deconv4_1 is {}".format(deconv4_1.shape))

        deconv3 = self._deconv_layer(deconv4_1, 128, 96, (4, 4), 2,
                                     out_shape=tf.shape(conv3), scope="deconv3")
        concat3 = tf.concat([conv3, deconv3], axis=3)
        deconv3_1 = self._conv_layer(concat3, 192, 96, (3, 3), 1, scope="deconv3_1")
        tf.logging.info("shape of deconv3_1 is {}".format(deconv3_1.shape))

        deconv2 = self._deconv_layer(deconv3_1, 96, 64, (4, 4), 2,
                                     out_shape=tf.shape(conv2), scope="deconv2")
        concat2 = tf.concat([conv2, deconv2], axis=3)
        deconv2_1 = self._conv_layer(concat2, 128, 64, (3, 3), 1, scope="deconv2_1")
        tf.logging.info("shape of deconv2_1 is {}".format(deconv2_1.shape))

        deconv1 = self._deconv_layer(deconv2_1, 64, 48, (4, 4), 2,
                                     out_shape=tf.shape(conv1), scope="deconv1")
        concat1 = tf.concat([conv1, deconv1], axis=3)
        deconv1_1 = self._conv_layer(concat1, 96, 48, (3, 3), 1, scope="deconv1_1")
        tf.logging.info("shape of deconv1_1 is {}".format(deconv1_1.shape))

        # self.deconv0 = self._deconv_layer(deconv1_1, 48, 12, (4, 4), 2,
        #                              out_shape=[tf.shape(conv0)[0], conv0.shape[1], conv0.shape[2], 12],
        #                              use_activation=False,
        #                              use_batch_norm=False,
        #                              scope="deconv0")
        # tf.logging.info("shape of deconv0 is {}".format(self.deconv0.shape))
        # tf.gather and tf.split are not supported in tensorRT-4, so calculate each output separately
        self.category_logits = self._deconv_layer(
            deconv1_1, 48, 1, (4, 4), 2,
            out_shape=[tf.shape(conv0)[0], conv0.shape[1], conv0.shape[2], 1],
            use_activation=False,
            use_batch_norm=False,
            scope="category_logits"
        )
        self.all_category_score = tf.nn.sigmoid(self.category_logits)
        self.category_score = tf.multiply(self.all_category_score, mask)

        self.instance = self._deconv_layer(
            deconv1_1, 48, 2, (4, 4), 2,
            out_shape=[tf.shape(conv0)[0], conv0.shape[1], conv0.shape[2], 2],
            use_activation=False,
            use_batch_norm=False,
            scope="instance"
        )

        self.confidence_logits = self._deconv_layer(
            deconv1_1, 48, 1, (4, 4), 2,
            out_shape=[tf.shape(conv0)[0], conv0.shape[1], conv0.shape[2], 1],
            use_activation=False,
            use_batch_norm=False,
            scope="confidence_logits"
        )
        self.all_confidence_score = tf.nn.sigmoid(self.confidence_logits)
        self.confidence_score = tf.multiply(self.all_confidence_score, mask)

        self.classify_logits = self._deconv_layer(
            deconv1_1, 48, 5, (4, 4), 2,
            out_shape=[tf.shape(conv0)[0], conv0.shape[1], conv0.shape[2], 5],
            use_activation=False,
            use_batch_norm=False,
            scope="classify_logits"
        )
        self.class_score = tf.nn.softmax(self.classify_logits)

        self.heading = self._deconv_layer(
            deconv1_1, 48, 2, (4, 4), 2,
            out_shape=[tf.shape(conv0)[0], conv0.shape[1], conv0.shape[2], 2],
            use_activation=False,
            use_batch_norm=False,
            scope="heading"
        )
        
        self.height = self._deconv_layer(
            deconv1_1, 48, 1, (4, 4), 2,
            out_shape=[tf.shape(conv0)[0], conv0.shape[1], conv0.shape[2], 1],
            use_activation=False,
            use_batch_norm=False,
            scope="height"
        )
    
    def _compute_loss(self, inputs, labels, mask, example_ids):
        # random set values of masked grids, so grids without LiDAR points can be scored near 0 by the scorer
        rand_unmask = CommonUtil.random_set(
            tf.shape(mask),
            ratio=self._config.unmask_ratio,
            value=1,
            set_area=1-mask
        )
        ext_mask = mask + rand_unmask

        raw_cat_score_lbl, instance_lbl, raw_conf_score_lbl, cls_score_lbl, heading_lbl, height_lbl \
            = tf.split(labels, [1, 2, 1, 5, 2, 1], axis=-1, name="label_split")

        cat_score_lbl = tf.math.minimum(raw_cat_score_lbl / self._config.category_label_thr * 0.5, 1.0)
        conf_score_lbl = tf.math.minimum(raw_conf_score_lbl / self._config.confidence_label_thr * 0.5, 1.0)
        cls_score_lbl = tf.math.argmax(cls_score_lbl, axis=-1)
        cls_score_lbl = tf.one_hot(cls_score_lbl, depth=5)

        if self._config.category_loss.lower() == "focal_loss":
            self.category_score_loss = self._focal_loss(self.all_category_score, cat_score_lbl, ext_mask)
        elif self._config.category_loss.lower() == "distance_aware_focal_loss":
            distance = tf.gather(inputs, self._config.dis_chan_idx, axis=-1)
            self.category_score_loss = self._distance_aware_focal_loss(
                self.all_category_score, cat_score_lbl, distance, ext_mask)

        self.instance_loss = self._cat_label_masked_mean_squared_error(self.instance, instance_lbl, mask, cat_score_lbl)

        if self._config.confidence_loss.lower() == "focal_loss":
            self.confidence_score_loss = self._focal_loss(self.all_confidence_score, conf_score_lbl, ext_mask)
        elif self._config.confidence_loss.lower() == "distance_aware_focal_loss":
            distance = tf.gather(inputs, self._config.dis_chan_idx, axis=-1)
            self.confidence_score_loss = self._distance_aware_focal_loss(
                self.all_confidence_score, conf_score_lbl, distance, ext_mask)

        self.class_score_loss = self._softmax_cross_entropy_loss(self.classify_logits, cls_score_lbl, mask)

        self.heading_loss = self._cat_label_masked_mean_squared_error(self.heading, heading_lbl, mask, cat_score_lbl)

        self.height_loss = self._cat_label_masked_mean_squared_error(self.height, height_lbl, mask, cat_score_lbl)

        self.loss = self.category_score_loss * self._config.loss_weights[0] \
                  + self.instance_loss * self._config.loss_weights[1] \
                  + self.confidence_score_loss * self._config.loss_weights[2] \
                  + self.class_score_loss * self._config.loss_weights[3] \
                  + self.heading_loss * self._config.loss_weights[4] \
                  + self.height_loss * self._config.loss_weights[5]

        if self._config.enable_summary:
            tf.summary.scalar("category_score_loss", self.category_score_loss)
            tf.summary.scalar("instance_loss", self.instance_loss)
            tf.summary.scalar("confidence_score_loss", self.confidence_score_loss)
            tf.summary.scalar("class_score_loss", self.class_score_loss)
            tf.summary.scalar("heading_loss", self.heading_loss)
            tf.summary.scalar("height_loss", self.height_loss)

            tf.summary.histogram("category_score_dist", self.category_score)
            tf.summary.histogram("confidence_score_dist", self.confidence_score)
            tf.summary.histogram("instance_dist", self.instance)

            # tf.summary.histogram("category_labels", cat_score_lbl)
            # tf.summary.histogram("instance_lables", instance_lbl)
            # tf.summary.histogram("confidence_score_labels", conf_score_lbl)
            # tf.summary.histogram("class_score_labels", cls_score_lbl)
            # tf.summary.histogram("heading_labels", heading_lbl)
            # tf.summary.histogram("height_labels", height_lbl)

            tf.summary.image("all_points_image", mask)
            tf.summary.image("category_pred_image", tf.to_float(tf.greater_equal(self.category_score, 0.5)))
            tf.summary.image("category_label_image", tf.to_float(tf.greater_equal(cat_score_lbl, 0.5)))
            tf.summary.image("confidence_pred_image", tf.to_float(tf.greater_equal(self.confidence_score, 0.5)))
            tf.summary.image("confidence_label_image", tf.to_float(tf.greater_equal(conf_score_lbl, 0.5)))
            tf.summary.text("example_ids", tf.strings.as_string(example_ids))

    def _compute_metric(self, labels, mask):
        cat_score_lbl, instance_lbl, conf_score_lbl, cls_score_lbl, heading_lbl, height_lbl \
            = tf.split(labels, [1, 2, 1, 5, 2, 1], axis=-1, name="label_split")

        cat_pred_class = tf.to_int32(tf.greater_equal(self.category_score, 0.5))
        cat_lbl_class = tf.to_int32(tf.greater_equal(cat_score_lbl, self._config.category_label_thr))
        self.category_precision = tf.metrics.precision(cat_lbl_class, cat_pred_class)
        self.category_recall = tf.metrics.recall(cat_lbl_class, cat_pred_class)

        self.instance_rmse = self._root_mean_squared_error_metric(self.instance, instance_lbl, mask)

        conf_pred_class = tf.to_int32(tf.greater_equal(self.confidence_score, 0.5))
        conf_lbl_class = tf.to_int32(tf.greater_equal(conf_score_lbl, self._config.confidence_label_thr))
        self.confidence_precision = tf.metrics.precision(conf_lbl_class, conf_pred_class)
        self.confidence_recall = tf.metrics.recall(conf_lbl_class, conf_pred_class)

        self.classify_iou = self._mean_iou_metric(self.class_score, cls_score_lbl, num_classes=5)

        self.heading_rmse = self._root_mean_squared_error_metric(self.heading, heading_lbl, mask)

        self.height_rmse = self._root_mean_squared_error_metric(self.height, height_lbl, mask)

    def get_var_assign_map(self, ckpt_dir):
        return {self.VARIABLE_SCOPE+"/": self.VARIABLE_SCOPE+"/"}
