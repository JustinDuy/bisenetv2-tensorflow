#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 03.01.2021
# @Author  : Justin Duy
# @Site    : https://github.com/JustinDuy/lanenet-lane-detection
# @File    : bisenet_keras_v2.py
# @IDE: PyCharm
"""
BiseNet V2 Model
"""
from semantic_segmentation_zoo import cnn_basenet
  
import tensorflow as tf
from tensorflow import keras
from keras.applications.xception import Xception,preprocess_input
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Conv2D, DepthwiseConv2D, Input, Dense, Dropout, Multiply, Dot, Concatenate, Add, GlobalAveragePooling2D, MaxPooling2D, concatenate
from keras.layers import BatchNormalization, Activation, AveragePooling2D, UpSampling2D
from keras.layers.experimental.preprocessing import Resizing
from keras.layers.core import Lambda
from keras.backend import tf as ktf
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard

class ConvBlk(keras.layers.Layer):
    """
    implementation of convolution block: CONV (-> Batch Norm -> Activation)
    """
    def __init__(self):
        """
        """
        super(ConvBlk, self).__init__()

    def call(self, input, **kwargs):
        #output_channels=32, k_size=(3,3), padding="SAME", stride=2, use_bias=False, activation="relu", need_activate=False
        output_channels = kwargs['output_channels']
        assert isinstance(output_channels, int) or (isinstance(output_channels[0], int) and isinstance(output_channels[1], int))

        self._conv = Conv2D(filters=self._output_channels, kernel_size=self._kernel, padding=self._padding, strides=self._stride, use_bias=self._use_bias)
        self._bn = BatchNormalization()
        self._act = Activation(self._activation)

        k_size = kwargs['k_size']
        assert isinstance(k_size, int)
        stride = kwargs['stride']
        assert isinstance(stride, int)

        if 'padding' in kwargs:
            self._padding = kwargs['padding']
        else:
            self._padding = "SAME"
        if 'use_bias' in kwargs:
            self._use_bias = kwargs['use_bias']
        else:
            self._use_bias = False
        if 'activation' in kwargs:
            self._activation = kwargs['activation']
        else:
            self._activation = "relu"
        if 'need_activate' in kwargs:
            self._need_activate = kwargs['need_activate']
        else:
            self._need_activate = False
        x = self._conv(input)
        x = self._bn(x)
        if self._need_activate:
            x = self._act(x)
        return x

class _StemBlock(keras.layers.Layer):
    """
    implementation of stem block module
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_StemBlock, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = "SAME"
    
    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def call(self, input, **kwargs):
        """
        :param args:
        :param kwargs:
        :return:
        """
        input_tensor = input
        output_channels = kwargs['output_channels']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']

        x = ConvBlk()(
            input_tensor, 
            output_channels=output_channels, 
            k_size=(3, 3), 
            stride=2)
        left_branch = ConvBlk()(
            x,
            output_channels=int(output_channels/2), 
            k_size=(1,1), 
            stride=1, 
            need_activate=True)
        left_branch = ConvBlk()(
            left_branch,
            output_channels=output_channels, 
            k_size=(3,3), 
            stride=2, 
            need_activate=True)
        right_branch = MaxPooling2D(pool_size=(3, 3), stride=2, need_activate=True)(x)
        concat = concatenate([left_branch, right_branch], axis=-1)
        result = ConvBlk()(
            concat,
            output_channels=output_channels, 
            k_size=(3, 3), 
            stride=1, 
            need_activate=True)
        return result

class _ContextEmbedding(keras.layers.Layer):
    """
    implementation of context embedding module in bisenetv2
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_ContextEmbedding, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = "SAME"
    
    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def call(self, inputs):
        """

        :param inputs:
        :return:
        """
        x = GlobalAveragePooling2D()(inputs)
        x = BatchNormalization()(x)
        x = ConvBlk()(
            x,
            output_channels=input_shape[-1], 
            k_size=(1, 1), 
            stride=1, 
            need_activate=True)
        fused = Add()[x, inputs]
        result = ConvBlk()(
            fused,
            output_channels=input_shape[-1], 
            k_size=(3, 3), 
            stride=2, 
            need_activate=True)
        return result

class _GatherExpansion(keras.layers.Layer):
    """
    implementation of gather and expansion module in bisenetv2
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_GatherExpansion, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = "SAME"
        self._stride = 1
        self._expansion_factor = 6

    def call(self, input, **kwargs):
        if 'padding' in kwargs:
            self._padding = kwargs['padding']
        if 'stride' in kwargs:
            self._stride = kwargs['stride']
        if 'e' in kwargs:
            self._expansion_factor = kwargs['e']
        if 'output_channels' in kwargs:
            output_channels = kwargs['output_channels']

        if self._stride == 1:
            x = ConvBlk()(
                input,
                output_channels=input_tensor_channels, 
                k_size=(3,3), 
                stride=1, 
                padding=self._padding, 
                need_activate=True)
            x = DepthwiseConv2D(
                kernel_size=(3,3),
                strides=1,
                padding=self._padding,
                depth_multiplier=self._expansion_factor)(x)
            x = BatchNormalization(x)
            x = ConvBlk()(
                x,
                output_channels=input_tensor_channels, 
                k_size=(1,1), 
                stride=1, 
                padding=self._padding, 
                need_activate=False)
            fused_features = Add()[input, x]
            result = self._relu(fused_features)
            return result
        elif self._stride == 2:
            input_proj = DepthwiseConv2D(
                kernel_size=(3,3),
                strides=self._stride,
                padding=self._padding,
                depth_multiplier=1)(input)
            input_proj = BatchNormalization(input_proj)
            input_proj = ConvBlk()(
                input_proj,
                output_channels=self._output_channels, 
                k_size=(1,1), 
                stride=1, 
                padding=self._padding, 
                need_activate=False)
            result = ConvBlk()(
                input,
                output_channels=input_tensor_channels, 
                k_size=(3,3), 
                stride=1, 
                padding=self._padding, 
                need_activate=True)
            result = DepthwiseConv2D(
                kernel_size=(3,3),
                strides=2,
                padding=self._padding,
                depth_multiplier=self._expansion_factor)(result)
            result = BatchNormalization(result)
            result = DepthwiseConv2D(
                kernel_size=(3,3),
                strides=1,
                padding=self._padding,
                depth_multiplier=1)(result)
            result = BatchNormalization(result)
            result = ConvBlk()(
                result,
                output_channels=self._output_channels, 
                k_size=(1,1), 
                stride=1, 
                padding=self._padding, 
                need_activate=False)
            fused_features = Add()[input_proj, result]
            result = Activation("relu")(fused_features)
            return result
        else:
            raise NotImplementedError('No function matched with stride of {}'.format(self._stride))

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

class _GuidedAggregation(keras.layers.Layer):
    """
    implementation of guided aggregation module in bisenetv2
    """

    def __init__(self, phase):
        """

        :param phase:
        """
        super(_GuidedAggregation, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = "SAME"

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def build(self, input_shapes):
        detail_input_tensor_shape = input_shapes[0]
        output_channels = detail_input_tensor_shape.as_list()[-1]
        self._detail_branch_3x3_dw_conv_block = DepthwiseConv2D(
            kernel_size=(3,3),
            strides=1,
            padding=self._padding,
            depth_multiplier=1)
        self._detail_branch_bn_1 = BatchNormalization()
        self._detail_branch_1x1_conv_block = Conv2D(
            filters=output_channels,
            kernel_size=1,
            padding=self._padding, 
            strides=1, 
            use_bias=False)
        self._detail_branch_3x3_conv_block = ConvBlk()
        self._detail_branch_avg_pooling_block = AveragePooling2D(
            pool_size=(3, 3),
            padding="SAME",
            strides=2)
        self._semantic_branch_3x3_dw_conv_block = DepthwiseConv2D(
            k_size=(3,3),
            stride=1,
            padding=self._padding,
            depth_multiplier=1)
        self._semantic_branch_bn_1 = BatchNormalization()
        self._semantic_branch_1x1_conv_block = Conv2D(
            filters=output_channels,
            kernel_size=1,
            padding=self._padding, 
            strides=1, 
            use_bias=False)
        self._semantic_branch_remain_sigmoid = Activation("sigmoid")
        self._semantic_branch_3x3_conv_block = ConvBlk()
        self._semantic_branch_upsample_features = Resizing(
            height=detail_input_tensor_shape[1],
            width=detail_input_tensor_shape[2],
            interpolation="bilinear")
        self._semantic_branch_upsample_sigmoid= Activation("sigmoid")

        self._guided_upsample_features = Resizing(
            height=detail_input_tensor_shape[1],
            width=detail_input_tensor_shape[2],
            interpolation="bilinear")
        self._aggregation_feature_conv_blk = ConvBlk()
        super(_GuidedAggregation, self).build(input_shape)  

    def call(self, inputs):
        """

        :param args:
        :param inputs:
        :return:
        """
        assert isinstance(inputs, list), "Expect list of tensors"

        detail_input_tensor = inputs[0]
        semantic_input_tensor = inputs[1]
        output_channels = detail_input_tensor.get_shape().as_list()[-1]

        # detail branch
        detail_branch_remain = self._detail_branch_3x3_dw_conv_block(detail_input_tensor)
        detail_branch_remain = self._detail_branch_bn_1(detail_branch_remain)
        detail_branch_remain = self._detail_branch_1x1_conv_block(detail_branch_remain)
        detail_branch_downsample = self._detail_branch_3x3_conv_block(
            detail_input_tensor,
            output_channels=output_channels, 
            k_size=(3,3), 
            stride=2, 
            padding=self._padding, 
            need_activate=False)
        detail_branch_downsample = self._detail_branch_avg_pooling_block(detail_branch_downsample)

        # semantic branch
        semantic_branch_remain = self._semantic_branch_3x3_dw_conv_block(semantic_input_tensor)
        semantic_branch_remain = self._semantic_branch_bn_1(semantic_branch_remain)
        semantic_branch_remain = self._semantic_branch_1x1_conv_block(semantic_branch_remain)
        semantic_branch_remain = self._semantic_branch_remain_sigmoid(semantic_branch_remain)
        semantic_branch_upsample = self._semantic_branch_3x3_conv_block(
            semantic_input_tensor,
            output_channels=output_channels, 
            k_size=(3,3), 
            stride=1, 
            padding=self._padding, 
            need_activate=False)
        semantic_branch_upsample = self._semantic_branch_upsample_features(semantic_branch_upsample)
        semantic_branch_upsample = self._semantic_branch_upsample_sigmoid(semantic_branch_upsample)

        # aggregation features
        guided_detail_features = Multiply()([detail_branch_remain, semantic_branch_upsample])
        guided_semantic_features = Multiply()([detail_branch_downsample, semantic_branch_remain])
        guided_upsample_features = self._guided_upsample_features(guided_semantic_features)
        fused_features = Add()([guided_detail_features, guided_upsample_features])
        aggregation_feature_output = self._aggregation_feature_conv_blk(
            fused_features,
            output_channels=detail_input_tensor_shape, 
            k_size=(3,3), 
            stride=1, 
            padding=self._padding, 
            need_activate=True)
        return aggregation_feature_output

class _SegmentationHead(keras.layers.Layer):
    """
    implementation of segmentation head in bisenet v2
    """
    def __init__(self, phase):
        """

        """
        super(_SegmentationHead, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = "SAME"

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def call(self, input, **kwargs):
        """

        :param args:
        :param inputs:
        :return:
        """
        upsample_ratio = kwargs['upsample_ratio']
        feature_dims = kwargs['feature_dims']
        classes_nums = kwargs['classes_nums']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']

        input_tensor_size = input.get_shape().as_list()[1:3]
        output_tensor_size = [int(tmp * ratio) for tmp in input_tensor_size]
        result = ConvBlk()(
            input_tensor,
            output_channels=self._feature_dims, 
            k_size=(3,3), 
            stride=1, 
            padding=self._padding, 
            need_activate=True)
        result = Conv2D(
            filters=self._classes_nums,
            kernel_size=1,
            padding=self._padding, 
            strides=1, 
            use_bias=False)(result)
        result = Resizing(
            height=output_tensor_size[0],
            width=output_tensor_size[1],
            interpolation="bilinear")(result)
        return result


class BiseNetV2(Model):
    """
    implementation of bisenet v2
    """
    def __init__(self, phase, cfg):
        """

        """
        super(BiseNetV2, self).__init__()
        self._cfg = cfg
        self._phase = phase
        self._is_training = self._is_net_for_training()

        # set model hyper params
        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY
        self._loss_type = self._cfg.SOLVER.LOSS_TYPE
        self._enable_ohem = self._cfg.SOLVER.OHEM.ENABLE
        if self._enable_ohem:
            self._ohem_score_thresh = self._cfg.SOLVER.OHEM.SCORE_THRESH
            self._ohem_min_sample_nums = self._cfg.SOLVER.OHEM.MIN_SAMPLE_NUMS
        self._ge_expand_ratio = self._cfg.MODEL.BISENETV2.GE_EXPAND_RATIO
        self._semantic_channel_ratio = self._cfg.MODEL.BISENETV2.SEMANTIC_CHANNEL_LAMBDA
        self._seg_head_ratio = self._cfg.MODEL.BISENETV2.SEGHEAD_CHANNEL_EXPAND_RATIO

        # set module used in bisenetv2
        self._conv_block = ConvBlk()
        self._se_block = _StemBlock(phase=phase)
        self._context_embedding_block = _ContextEmbedding(phase=phase)
        self._ge_block = _GatherExpansion(phase=phase)
        self._guided_aggregation_block = _GuidedAggregation(phase=phase)
        self._seg_head_block = _SegmentationHead(phase=phase)

        # set detail branch channels
        self._detail_branch_channels = self._build_detail_branch_hyper_params()
        # set semantic branch channels
        self._semantic_branch_channels = self._build_semantic_branch_hyper_params()

        # set op block params
        self._block_maps = {
            'conv_block': self._conv_block,
            'se': self._se_block,
            'ge': self._ge_block,
            'ce': self._context_embedding_block,
        }

        self._net_intermediate_results = collections.OrderedDict()

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    @classmethod
    def _build_detail_branch_hyper_params(cls):
        """

        :return:
        """
        params = [
            # stage        op           k  c   s  r
            ('stage_1', [('conv_block', 3, 64, 2, 1), ('conv_block', 3, 64, 1, 1)]),
            ('stage_2', [('conv_block', 3, 64, 2, 1), ('conv_block', 3, 64, 1, 2)]),
            ('stage_3', [('conv_block', 3, 128, 2, 1), ('conv_block', 3, 128, 1, 2)]),
        ]
        return collections.OrderedDict(params)

    def _build_semantic_branch_hyper_params(self):
        """

        :return:
        """
        stage_1_channels = int(self._detail_branch_channels['stage_1'][0][2] * self._semantic_channel_ratio)
        stage_3_channels = int(self._detail_branch_channels['stage_3'][0][2] * self._semantic_channel_ratio)
        params = [
            ('stage_1', [('se', 3, stage_1_channels, 1, 4, 1)]),
            ('stage_3', [('ge', 3, stage_3_channels, self._ge_expand_ratio, 2, 1),
                         ('ge', 3, stage_3_channels, self._ge_expand_ratio, 1, 1)]),
            ('stage_4', [('ge', 3, stage_3_channels * 2, self._ge_expand_ratio, 2, 1),
                         ('ge', 3, stage_3_channels * 2, self._ge_expand_ratio, 1, 1)]),
            ('stage_5', [('ge', 3, stage_3_channels * 4, self._ge_expand_ratio, 2, 1),
                         ('ge', 3, stage_3_channels * 4, self._ge_expand_ratio, 1, 3),
                         ('ce', 3, stage_3_channels * 4, self._ge_expand_ratio, 1, 1)])
        ]
        return collections.OrderedDict(params)

    def build_detail_branch(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        result = input_tensor
        for stage_name, stage_params in self._detail_branch_channels.items():
            for block_index, param in enumerate(stage_params):
                block_op_name = param[0]
                block_op = self._block_maps[block_op_name]
                assert block_op_name == 'conv_block'
                k_size = param[1]
                output_channels = param[2]
                stride = param[3]
                repeat_times = param[4]
                for repeat_index in range(repeat_times):
                    if stage_name == 'stage_3' and block_index == 1 and repeat_index == 1:
                        result = ConvBlk()
                        (result, 
                         k_size=k_size,
                         output_channels=output_channels,
                         stride=stride,
                         padding="SAME",
                         use_bias=False,
                         need_activate=False
                        )
                    else:
                        result = block_op(
                            k_size=k_size,
                            output_channels=output_channels,
                            stride=stride,
                            padding="SAME",
                            use_bias=False,
                            need_activate=True
                        )(result)
        return result

    def build_semantic_branch(self, input_tensor, name, prepare_data_for_booster=False):
        """

        :param input_tensor:
        :param name:
        :param prepare_data_for_booster:
        :return:
        """
        seg_head_inputs = collections.OrderedDict()
        result = input_tensor
        source_input_tensor_size = input_tensor.get_shape().as_list()[1:3]
        for stage_name, stage_params in self._semantic_branch_channels.items():
            seg_head_input = input_tensor
            for block_index, param in enumerate(stage_params):
                block_op_name = param[0]
                block_op = self._block_maps[block_op_name]
                output_channels = param[2]
                expand_ratio = param[3]
                stride = param[4]
                repeat_times = param[5]
                for repeat_index in range(repeat_times):
                    if block_op_name == 'ge':
                        result = block_op(
                            result, 
                            stride=stride, 
                            e=expand_ratio, 
                            output_channels=output_channels
                        )
                        seg_head_input = result
                    elif block_op_name == 'ce':
                        result = block_op(
                            result
                        )
                    elif block_op_name == 'se':
                        result = block_op(
                            input_tensor=result,
                            output_channels=output_channels,
                            name='stem_block'
                        )
                        seg_head_input = result
                    else:
                        raise NotImplementedError('Not support block type: {:s}'.format(block_op_name))
            if prepare_data_for_booster:
                result_tensor_size = result.get_shape().as_list()[1:3]
                result_tensor_dims = result.get_shape().as_list()[-1]
                upsample_ratio = int(source_input_tensor_size[0] / result_tensor_size[0])
                feature_dims = result_tensor_dims * self._seg_head_ratio
                seg_head_inputs[stage_name] = self._seg_head_block(
                    input_tensor=seg_head_input,
                    name='block_{:d}_seg_head_block'.format(block_index + 1),
                    upsample_ratio=upsample_ratio,
                    feature_dims=feature_dims,
                    classes_nums=self._class_nums
                    )
        return result, seg_head_inputs

    def build_aggregation_branch(self, detail_output, semantic_output, name):
        """

        :param detail_output:
        :param semantic_output:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self._guided_aggregation_block(
                detail_input_tensor=detail_output,
                semantic_input_tensor=semantic_output,
                name='guided_aggregation_block'
            )
        return result

    def build_instance_segmentation_branch(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        input_tensor_size = input_tensor.get_shape().as_list()[1:3]
        output_tensor_size = [int(tmp * 8) for tmp in input_tensor_size]

        with tf.variable_scope(name_or_scope=name):
            output_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=64,
                stride=1,
                name='conv_3x3',
                use_bias=False,
                need_activate=True
            )
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=128,
                stride=1,
                name='conv_1x1',
                use_bias=False,
                need_activate=False
            )
            output_tensor = tf.image.resize_bilinear(
                output_tensor,
                output_tensor_size,
                name='instance_logits'
            )
        return output_tensor

    def build_binary_segmentation_branch(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        input_tensor_size = input_tensor.get_shape().as_list()[1:3]
        output_tensor_size = [int(tmp * 8) for tmp in input_tensor_size]

        with tf.variable_scope(name_or_scope=name):
            output_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=64,
                stride=1,
                name='conv_3x3',
                use_bias=False,
                need_activate=True
            )
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=128,
                stride=1,
                name='conv_1x1',
                use_bias=False,
                need_activate=True
            )
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=self._class_nums,
                stride=1,
                name='final_conv',
                use_bias=False,
                need_activate=False
            )
            output_tensor = tf.image.resize_bilinear(
                output_tensor,
                output_tensor_size,
                name='binary_logits'
            )
        return output_tensor

    def build_model(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # build detail branch
            detail_branch_output = self.build_detail_branch(
                input_tensor=input_tensor,
                name='detail_branch'
            )
            # build semantic branch
            semantic_branch_output, _ = self.build_semantic_branch(
                input_tensor=input_tensor,
                name='semantic_branch',
                prepare_data_for_booster=False
            )
            # build aggregation branch
            aggregation_branch_output = self.build_aggregation_branch(
                detail_output=detail_branch_output,
                semantic_output=semantic_branch_output,
                name='aggregation_branch'
            )
            # build binary and instance segmentation branch
            binary_seg_branch_output = self.build_binary_segmentation_branch(
                input_tensor=aggregation_branch_output,
                name='binary_segmentation_branch'
            )
            instance_seg_branch_output = self.build_instance_segmentation_branch(
                input_tensor=aggregation_branch_output,
                name='instance_segmentation_branch'
            )
            # gather frontend output result
            self._net_intermediate_results['binary_segment_logits'] = {
                'data': binary_seg_branch_output,
                'shape': binary_seg_branch_output.get_shape().as_list()
            }
            self._net_intermediate_results['instance_segment_logits'] = {
                'data': instance_seg_branch_output,
                'shape': instance_seg_branch_output.get_shape().as_list()
            }
        return self._net_intermediate_results


if __name__ == '__main__':
    """
    test code
    """
    test_in_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    model = BiseNetV2(phase='train', cfg=parse_config_utils.lanenet_cfg)
    ret = model.build_model(test_in_tensor, name='bisenetv2')
    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))

