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
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
#from keras.applications.xception import Xception,preprocess_input
#from keras.optimizers import SGD
import collections
from keras.models import Model
from keras.layers import Conv2D, DepthwiseConv2D, Input, Dense, Dropout, Multiply, Dot, Concatenate, Add, GlobalAveragePooling2D, MaxPooling2D, concatenate
from keras.layers import BatchNormalization, Activation, AveragePooling2D, UpSampling2D
from keras.layers.experimental.preprocessing import Resizing
from keras.layers.core import Lambda
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard

class ConvBlk(keras.layers.Layer):
    """
    implementation of convolution block: CONV (-> Batch Norm -> Activation)
    """
    def __init__(self, output_channels, k_size, stride, padding="SAME", use_bias=False, activation="relu", need_activate=False):
        """
        """
        self.output_channels = output_channels
        assert isinstance(self.output_channels, int) or (isinstance(self.output_channels[0], int) and isinstance(self.output_channels[1], int))
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.activation = activation
        self.need_activate = need_activate
        super(ConvBlk, self).__init__()

    def build(self, input_shape):
        self._conv = Conv2D(
            filters=self.output_channels,
            kernel_size=self.k_size,
            padding=self.padding,
            strides=self.stride,
            use_bias=self.use_bias
        )
        self._bn = BatchNormalization()
        self._act = Activation(self.activation)
        super(ConvBlk, self).build(input_shape)

    def call(self, input):

        x = self._conv(input)
        x = self._bn(x)
        if self.need_activate:
            x = self._act(x)
        return x

class _StemBlock(keras.layers.Layer):
    """
    implementation of stem block module
    """
    def __init__(self, phase, output_channels, padding = "SAME"):
        """

        :param phase:
        """
        super(_StemBlock, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._output_channels = output_channels
        self._padding = padding

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

    def build(self, input_shape):
        self._conv_3x3_1 = ConvBlk(
            output_channels=self._output_channels,
            k_size=3,
            stride=2)
        self._conv_1x1 = ConvBlk(
            output_channels=int(self._output_channels/2),
            k_size=1,
            stride=1,
            need_activate=True)
        self._conv_3x3_2 = ConvBlk(
            output_channels=self._output_channels,
            k_size=3,
            stride=2,
            need_activate=True)
        self._max_pool = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same')
        self._conv_3x3_3 = ConvBlk(
            output_channels=self._output_channels,
            k_size=3,
            stride=1,
            need_activate=True)
        super(_StemBlock, self).build(input_shape)

    def call(self, input):
        """
        :param args:
        :param kwargs:
        :return:
        """
        x = self._conv_3x3_1(input)
        branch_left_output = self._conv_1x1(x)
        branch_left_output = self._conv_3x3_2(branch_left_output)
        branch_right_output = self._max_pool(x)
        concat = concatenate([branch_left_output, branch_right_output], axis=-1)
        result = self._conv_3x3_3(concat)
        return result

class _ContextEmbedding(keras.layers.Layer):
    """
    implementation of context embedding module in BiseNetKerasV2
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

    def build(self, input_shape):
        output_channels = input_shape.as_list()[-1]
        self._global_avg_pool = GlobalAveragePooling2D()
        self._bn = BatchNormalization()
        self._conv_1x1 = ConvBlk(
            output_channels=output_channels,
            k_size=1,
            stride=1,
            need_activate=True)
        self._add = Add()
        self._conv_3x3 = ConvBlk(
            output_channels=output_channels,
            k_size=(3, 3),
            stride=1,
            need_activate=True)
        super(_ContextEmbedding, self).build(input_shape)

    def call(self, input):
        """

        :param input:
        :return:
        """
        #x = self._global_avg_pool(input)
        x = tf.math.reduce_mean(input, axis=[1, 2], keepdims=True)
        x = self._bn(x)
        x = self._conv_1x1(x)
        fused = self._add ([x, input])
        result = self._conv_3x3(fused)
        assert result.get_shape().as_list()[1] == input.get_shape().as_list()[1]
        assert result.get_shape().as_list()[2] == input.get_shape().as_list()[2]
        return result

class _GatherExpansion(keras.layers.Layer):
    """
    implementation of gather and expansion module in BiseNetKerasV2
    """
    def __init__(self, phase, output_channels, padding="SAME", stride=1, e=6):
        """

        :param phase:
        """
        super(_GatherExpansion, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()

        self._padding = padding
        self._stride = stride
        self._expansion_factor = e
        self._output_channels = output_channels

    def build(self, input_shape):
        input_tensor_channels = input_shape.as_list()[-1]
        self._stride1_conv_3x3 = ConvBlk(
            output_channels=input_tensor_channels,
            k_size=3,
            stride=1,
            padding=self._padding,
            need_activate=True)
        self._stride1_deepwise_conv = DepthwiseConv2D(
                kernel_size=(3,3),
                strides=1,
                padding=self._padding,
                depth_multiplier=self._expansion_factor)
        self._stride1_bn = BatchNormalization()
        self._stride1_conv_1x1 = ConvBlk(
                output_channels=input_tensor_channels,
                k_size=1,
                stride=1,
                padding=self._padding,
                need_activate=False)
        self._stride1_add = Add()
        self._stride1_act = Activation("relu")

        self._stride2_deepwise_conv1 = DepthwiseConv2D(
                kernel_size=(3,3),
                strides=self._stride,
                padding=self._padding,
                depth_multiplier=1)
        self._stride2_bn1 = BatchNormalization()
        self._stride2_conv_1x1 = ConvBlk(
                output_channels=self._output_channels,
                k_size=(1,1),
                stride=1,
                padding=self._padding,
                need_activate=False)
        self._stride2_conv_3x3 =ConvBlk(
                output_channels=input_tensor_channels,
                k_size=(3,3),
                stride=1,
                padding=self._padding,
                need_activate=True)
        self._stride2_deepwise_conv2 = DepthwiseConv2D(
                kernel_size=(3,3),
                strides=2,
                padding=self._padding,
                depth_multiplier=self._expansion_factor)
        self._stride2_bn2 = BatchNormalization()
        self._stride2_deepwise_conv3 =DepthwiseConv2D(
                kernel_size=(3,3),
                strides=1,
                padding=self._padding,
                depth_multiplier=1)
        self._stride2_bn3 = BatchNormalization()
        self._stride2_conv1x1 = ConvBlk(
                output_channels=self._output_channels,
                k_size=(1,1),
                stride=1,
                padding=self._padding,
                need_activate=False)
        self._stride2_add = Add()
        self._stride2_act =Activation('relu')
        super(_GatherExpansion, self).build(input_shape)

    def call(self, input):
        input_tensor_channels = input.get_shape().as_list()[-1]
        if self._stride == 1:
            x = self._stride1_conv_3x3(input)
            x = self._stride1_deepwise_conv(x)
            x = self._stride1_bn(x)
            x = self._stride1_conv_1x1(x)
            fused_features = self._stride1_add([input, x])
            result = self._stride1_act(fused_features)
            assert result.get_shape().as_list()[1] == input.get_shape().as_list()[1]
            assert result.get_shape().as_list()[2] == input.get_shape().as_list()[2]
            return result
        elif self._stride == 2:
            input_proj = self._stride2_deepwise_conv1(input)
            input_proj = self._stride2_bn1(input_proj)
            input_proj = self._stride2_conv_1x1(input_proj)
            result = self._stride2_conv_3x3(input)
            result = self._stride2_deepwise_conv2(result)
            result = self._stride2_bn2(result)
            result = self._stride2_deepwise_conv3(result)
            result = self._stride2_bn3(result)
            result = self._stride2_conv1x1(result)
            fused_features = self._stride2_add([input_proj, result])
            result = self._stride2_act(fused_features)
            assert (result.get_shape().as_list()[1] * 2) == input.get_shape().as_list()[1]
            assert (result.get_shape().as_list()[2] * 2) == input.get_shape().as_list()[2]
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
    implementation of guided aggregation module in BiseNetKerasV2
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
        self._detail_branch_3x3_conv_block = ConvBlk(
            output_channels=output_channels,
            k_size=3,
            stride=2,
            padding=self._padding,
            need_activate=False)
        self._detail_branch_avg_pooling_block = AveragePooling2D(
            pool_size=(3, 3),
            padding="SAME",
            strides=2)
        self._semantic_branch_3x3_dw_conv_block = DepthwiseConv2D(
            kernel_size=(3,3),
            strides=1,
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
        self._semantic_branch_3x3_conv_block = ConvBlk(
            output_channels=output_channels,
            k_size=3,
            stride=1,
            padding=self._padding,
            need_activate=False)
        self._semantic_branch_upsample_sigmoid= Activation("sigmoid")
        self._aggregation_feature_conv_blk = ConvBlk(
            output_channels=output_channels,
            k_size=(3,3),
            stride=1,
            padding=self._padding,
            need_activate=True)
        super(_GuidedAggregation, self).build(input_shapes)  

    def call(self, inputs):
        """
        :param inputs:
        :return:
        """
        assert isinstance(inputs, list), "Expect list of input tensors"

        detail_input_tensor = inputs[0]
        semantic_input_tensor = inputs[1]

        output_channels = detail_input_tensor.get_shape().as_list()[-1]
        # detail branch
        detail_branch_remain = self._detail_branch_3x3_dw_conv_block(detail_input_tensor)
        detail_branch_remain = self._detail_branch_bn_1(detail_branch_remain)
        detail_branch_remain = self._detail_branch_1x1_conv_block(detail_branch_remain)
        detail_branch_downsample = self._detail_branch_3x3_conv_block(
            detail_input_tensor)
        detail_branch_downsample = self._detail_branch_avg_pooling_block(detail_branch_downsample)

        # semantic branch
        semantic_branch_remain = self._semantic_branch_3x3_dw_conv_block(semantic_input_tensor)
        semantic_branch_remain = self._semantic_branch_bn_1(semantic_branch_remain)
        semantic_branch_remain = self._semantic_branch_1x1_conv_block(semantic_branch_remain)
        semantic_branch_remain = self._semantic_branch_remain_sigmoid(semantic_branch_remain)
        semantic_branch_upsample = self._semantic_branch_3x3_conv_block(
            semantic_input_tensor
        )
        semantic_branch_upsample = tf.image.resize(
            semantic_branch_upsample,
            detail_input_tensor.shape[1:3],
            method='bilinear'
        )
        semantic_branch_upsample = self._semantic_branch_upsample_sigmoid(semantic_branch_upsample)

        # aggregation features
        guided_detail_features = tf.multiply(detail_branch_remain, semantic_branch_upsample)
        guided_semantic_features =  tf.multiply(detail_branch_downsample, semantic_branch_remain)
        guided_features_upsample = tf.image.resize(
            guided_semantic_features,
            detail_input_tensor.shape[1:3],
            method='bilinear'
        )
        fused_features = tf.add(guided_detail_features, guided_features_upsample)
        aggregation_feature_output = self._aggregation_feature_conv_blk(
            fused_features
        )
        return aggregation_feature_output

class _SegmentationHead(keras.layers.Layer):
    """
    implementation of segmentation head in bisenet v2
    """
    def __init__(self, phase, upsample_ratio, feature_dims, classes_nums, padding="SAME"):
        """

        """
        super(_SegmentationHead, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._upsample_ratio = upsample_ratio
        self._feature_dims = feature_dims
        self._classes_nums = classes_nums
        self._padding = padding

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

    def build(self, input_shape):
        input_tensor_size = input_shape.as_list()[1:3]
        output_tensor_size = [int(tmp * self._upsample_ratio) for tmp in input_tensor_size]
        self._conv3x3 = ConvBlk(
            output_channels=self._feature_dims,
            k_size=(3,3),
            stride=1,
            padding=self._padding,
            need_activate=True)
        self._conv1x1 = Conv2D(
            filters=self._classes_nums,
            kernel_size=1,
            padding=self._padding,
            strides=1,
            use_bias=False)
        self._bilinear_resize = Resizing(
            height=output_tensor_size[0],
            width=output_tensor_size[1],
            interpolation="bilinear")
        super(_SegmentationHead, self).build(input_shape)

    def call(self, input):
        """

        :param input:
        :return:
        """
        result = self._conv3x3(input)
        result = self._conv1x1(result)
        result = self._bilinear_resize(result)
        return result

class _DetailBranch(keras.layers.Layer):

    """
    implement bisenet v2 's detail branch
    """
    def __init__(self, phase):
        super(_DetailBranch, self).__init__()
        self.phase = phase
        self.params = self.get_params()
        self.stages = [{} for i in range(len(self.params))]

    def get_params(self):
        params = [
            # stage        opr          k  c   s  r
            ('stage_1', [('conv_block', 3, 64, 2, 1), ('conv_block', 3, 64, 1, 1)]),
            ('stage_2', [('conv_block', 3, 64, 2, 1), ('conv_block', 3, 64, 1, 2)]),
            ('stage_3', [('conv_block', 3, 128, 2, 1), ('conv_block', 3, 128, 1, 2)]),
        ]
        return collections.OrderedDict(params)

    def build(self, input_shape):
        stages_params = list(self.params.items())
        for stage_index, stage_blocks in enumerate(self.stages):
            stage_name, stage_params = stages_params[stage_index]
            self.stages[stage_index] = [{} for block_index in range(len(stage_params))]
            for block_index, params in enumerate(stage_params):
                block_op = params[0]
                assert block_op == 'conv_block'
                k_size = params[1]
                output_channels = params[2]
                stride = params[3]
                repeat_times = params[4]
                block_filters = [{} for repeat_index in range(repeat_times)]
                for repeat_index in range(repeat_times):
                    if stage_name == 'stage_3' and block_index == 1 and repeat_index == 1:
                        block_filters[repeat_index] = ConvBlk(
                            k_size=k_size,
                            output_channels=output_channels,
                            stride=stride,
                            padding="SAME",
                            use_bias=False,
                            need_activate=False
                        )
                    else:
                        block_filters[repeat_index] = ConvBlk(
                            k_size=k_size,
                            output_channels=output_channels,
                            stride=stride,
                            padding="SAME",
                            use_bias=False,
                            need_activate=True
                        )
                self.stages[stage_index][block_index] = block_filters

        super(_DetailBranch, self).build(input_shape)

    def call(self, input):
        result = input
        for stage_blocks in self.stages:
            for block in stage_blocks:
                for filter in block:
                    result = filter(result)
        return result

class _SemanticBranch(keras.layers.Layer):
    """
    implement bisenet v2 's semantic branch
    """
    def __init__(self, phase, semantic_channel_ratio, ge_expand_ratio, seg_head_ratio, class_nums):
        super(_SemanticBranch, self).__init__()
        self._phase = phase
        self._seg_head_ratio = seg_head_ratio
        self.params = self.get_params(semantic_channel_ratio, ge_expand_ratio)
        self.stages = [{} for stage in range(len(self.params))]
        self._seg_head = [{} for stage in range(len(self.params))]
        self._class_nums = class_nums

    def get_params(self, semantic_channel_ratio, ge_expand_ratio):
        detail_branch_params = _DetailBranch(phase="training").get_params()
        stage_1_channels = int(detail_branch_params['stage_1'][0][2] * semantic_channel_ratio)
        assert stage_1_channels == 16
        stage_3_channels = int(detail_branch_params['stage_3'][0][2] * semantic_channel_ratio)
        assert stage_3_channels == 32
        params = [
            ('stage_1', [('se', 3, stage_1_channels, 1, 4, 1, 2)]),
            ('stage_3', [('ge', 3, stage_3_channels, ge_expand_ratio, 2, 1, 8),
                         ('ge', 3, stage_3_channels, ge_expand_ratio, 1, 1, 8)]),
            ('stage_4', [('ge', 3, stage_3_channels * 2, ge_expand_ratio, 2, 1, 16),
                         ('ge', 3, stage_3_channels * 2, ge_expand_ratio, 1, 1, 16)]),
            ('stage_5', [('ge', 3, stage_3_channels * 4, ge_expand_ratio, 2, 1, 32),
                         ('ge', 3, stage_3_channels * 4, ge_expand_ratio, 1, 3, 32),
                         ('ce', 3, stage_3_channels * 4, 1, 1, 1, 32)])
        ]
        return collections.OrderedDict(params)

    def build(self, input_shape):
        source_input_tensor_size = input_shape.as_list()[1:3]
        stages_params = list(self.params.items())
        for stage_index, stage_blocks in enumerate(self.stages):
            stage_name, stage_params = stages_params[stage_index]
            self.stages[stage_index] = [{} for block_index in range(len(stage_params))]
            for block_index, params in enumerate(stage_params):
                block_op_name = params[0]
                output_channels = params[2]
                expand_ratio = params[3]
                stride = params[4]
                repeat_times = params[5]
                block_filters = [{} for repeat_index in range(repeat_times)]
                for repeat_index in range(repeat_times):
                    if block_op_name == 'ge':
                        block_filters[repeat_index] = _GatherExpansion(
                            phase = self._phase,
                            stride=stride,
                            e=expand_ratio,
                            output_channels=output_channels
                        )
                    elif block_op_name == 'ce':
                        block_filters[repeat_index] = _ContextEmbedding(self._phase)
                    elif block_op_name == 'se':
                        block_filters[repeat_index] = _StemBlock(self._phase, output_channels=output_channels)
                    else:
                        raise NotImplementedError('Not support block type: {:s}'.format(block_op_name))
                self.stages[stage_index][block_index] = block_filters

            result_tensor_dims = stage_params[-1][2]
            upsample_ratio = stage_params[-1][6]
            feature_dims = result_tensor_dims * self._seg_head_ratio
            self._seg_head[stage_index] = _SegmentationHead(
                phase=self._phase,
                upsample_ratio=upsample_ratio,
                feature_dims=feature_dims,
                classes_nums=self._class_nums
            )

        super(_SemanticBranch, self).build(input_shape)

    def call(self, input, prepare_data_for_booster=False):
        result = input
        seg_head_inputs = collections.OrderedDict()
        stages_params = list(self.params.items())
        for stage_index, stage_blocks in enumerate(self.stages):
            stage_name, stage_params = stages_params[stage_index]
            seg_head_input = input
            for block_index, params in enumerate(stage_params):
                block_filters = stage_blocks[block_index]
                block_op_name = params[0]
                for filter in block_filters:
                    result = filter(result)
                    if block_op_name == 'se':
                        seg_head_input = result
            if prepare_data_for_booster:
                seg_head_inputs[stage_name] = self._seg_head[stage_index](result)
        return result, seg_head_inputs

class _BinarySegBranch(keras.layers.Layer):
    """
    implement binary segmentation branch of Bisenetv2
    """
    def __init__(self):
        super(_BinarySegBranch, self).__init__()

    def build(self, input_shape):
        input_tensor_size = input_shape.as_list()[1:3]
        output_tensor_size = [int(tmp * 8) for tmp in input_tensor_size]

        self.conv3x3 = ConvBlk(
            k_size=3,
            output_channels=64,
            stride=1,
            use_bias=False,
            need_activate=True
        )
        self.conv1x1_1 = ConvBlk(
            k_size=1,
            output_channels=128,
            stride=1,
            use_bias=False,
            need_activate=True
        )
        self.conv1x1_2 = ConvBlk(
            k_size=1,
            output_channels=self._class_nums,
            stride=1,
            use_bias=False,
            need_activate=False
        )
        self.resize = Resizing(
            height=output_tensor_size[0],
            width=output_tensor_size[1],
            interpolation="bilinear")
        super(_BinarySegBranch, self).build(input_shape)

    def call(self, input):
        output_tensor = self.conv3x3(input)
        output_tensor = self.conv1x1_1(output_tensor)
        output_tensor = self.conv1x1_2(output_tensor)
        output_tensor = self.resize(output_tensor)
        return output_tensor

class BiseNetKerasV2(Model):
    """
    implementation of bisenet v2
    """
    def __init__(self, phase, cfg):
        """

        """
        super(BiseNetKerasV2, self).__init__()
        self._cfg = cfg
        self._phase = phase
        self._is_training = self._is_net_for_training()

        # set model hyper params
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY
        self._loss_type = self._cfg.SOLVER.LOSS_TYPE
        self._enable_ohem = self._cfg.SOLVER.OHEM.ENABLE
        if self._enable_ohem:
            self._ohem_score_thresh = self._cfg.SOLVER.OHEM.SCORE_THRESH
            self._ohem_min_sample_nums = self._cfg.SOLVER.OHEM.MIN_SAMPLE_NUMS
        self._ge_expand_ratio = self._cfg.MODEL.BISENETV2.GE_EXPAND_RATIO
        self._semantic_channel_ratio = self._cfg.MODEL.BISENETV2.SEMANTIC_CHANNEL_LAMBDA
        self._seg_head_ratio = self._cfg.MODEL.BISENETV2.SEGHEAD_CHANNEL_EXPAND_RATIO
        self._class_nums = self._cfg.DATASET.NUM_CLASSES

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
    def _compute_cross_entropy_loss(cls, seg_logits, labels, class_nums):
        """

        :param seg_logits:
        :param labels:
        :param class_nums:
        :return:
        """
        number_of_heads = seg_logits.shape[-1]
        np_array = [0.0 for _ in range(labels.shape[0])] # define init loss for whole batch
        loss_value = tf.constant(np_array)
        for head in range(number_of_heads):
            # first check if the logits' shape is matched with the labels'
            head_shape = seg_logits.shape.as_list()
            seg_head = tf.slice(seg_logits, [0, 0, 0, 0, 0], [head_shape[0], head_shape[1], head_shape[2], head_shape[3], 0])
            print("Head "  + head + " " + seg_head.shape + " " )
            seg_logits_shape = seg_head.shape[1:3]
            labels_shape = labels.shape[1:3]
            seg_logit = tf.cond(
                tf.reduce_all(tf.equal(seg_logits_shape, labels_shape)),
                true_fn=lambda: tf.dtypes.cast(seg_head, tf.float32),
                false_fn=lambda: tf.image.resize(seg_head, labels_shape, method='bilinear')
            )
            seg_logit = tf.reshape(seg_logit, [-1, class_nums])
            labels = tf.reshape(labels, [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(labels, class_nums - 1)), 1)
            seg_logit = tf.gather(seg_logit, indices)
            labels = tf.cast(tf.gather(labels, indices), tf.int32)

            # compute cross entropy loss
            loss_value += tf.math.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=seg_logit
                ),
                name='cross_entropy_loss'
            )
        #K.print_tensor(loss_value)
        return loss_value

    @classmethod
    def _compute_ohem_cross_entropy_loss(cls, seg_logits, labels, class_nums, thresh, n_min):
        """

        :param seg_logits:
        :param labels:
        :param class_nums:
        :return:
        """
        number_of_heads = seg_logits.shape[-1]
        np_array = [0.0 for i in range(labels.shape[0])]  # define init loss for whole batch
        loss_value = tf.constant(np_array)
        for head in range(number_of_heads):
            # first check if the logits' shape is matched with the labels'
            seg_logits_shape = seg_logits.shape[1:3]
            labels_shape = labels.shape[1:3]
            seg_logits = tf.cond(
                tf.reduce_all(tf.equal(seg_logits_shape, labels_shape)),
                true_fn=lambda: tf.dtypes.cast(seg_logits, tf.float32),
                false_fn=lambda: tf.image.resize(seg_logits, labels_shape, method='bilinear')
            )
            seg_logits = tf.reshape(seg_logits, [-1, class_nums])
            labels = tf.reshape(labels, [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(labels, class_nums - 1)), 1)
            seg_logits = tf.gather(seg_logits, indices)
            labels = tf.cast(tf.gather(labels, indices), tf.int32)

            # compute cross entropy loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=seg_logits
            )
            loss, _ = tf.nn.top_k(loss, tf.size(loss), sorted=True)

            # apply ohem
            ohem_thresh = tf.multiply(-1.0, tf.math.log(thresh), name='ohem_score_thresh')
            ohem_cond = tf.greater(loss[n_min], ohem_thresh)
            loss_select = tf.cond(
                pred=ohem_cond,
                true_fn=lambda: tf.gather(loss, tf.squeeze(tf.where(tf.greater(loss, ohem_thresh)), 1)),
                false_fn=lambda: loss[:n_min]
            )
            loss_value += tf.math.reduce_mean(loss_select, name='ohem_cross_entropy_loss')
        #K.print_tensor(loss_value)
        return loss_value

    @classmethod
    def _compute_l2_reg_loss(cls, var_list, weights_decay, name):
        """

        :param var_list:
        :param weights_decay:
        :param name:
        :return:
        """

        l2_reg_loss = tf.constant(0.0, tf.float32)
        for vv in var_list:
            if 'beta' in vv.name or 'gamma' in vv.name or 'b:0' in vv.name.split('/')[-1]:
                continue
            else:
                l2_reg_loss += tf.nn.l2_loss(vv)
        l2_reg_loss *= weights_decay
        l2_reg_loss = tf.identity(l2_reg_loss, 'l2_loss')
        #K.print_tensor(l2_reg_loss)
        return l2_reg_loss

    def build(self, input_shape):
        detail_params = _DetailBranch(self._phase).get_params()
        detail_output_channels = detail_params['stage_3'][-1][2]
        # detail branch
        self._detail_branch = _DetailBranch(self._phase)
        # semantic branch
        self._semantic_branch = _SemanticBranch(
            self._phase,
            semantic_channel_ratio = self._semantic_channel_ratio,
            ge_expand_ratio = self._ge_expand_ratio,
            seg_head_ratio = self._seg_head_ratio,
            class_nums = self._class_nums
        )
        # guided aggregation branch
        self._guided_aggregation_branch = _GuidedAggregation(self._phase)
        # sematic head branches
        self._semantic_heads = _SegmentationHead(
            phase='train',
            upsample_ratio=8,
            feature_dims=self._seg_head_ratio * detail_output_channels,
            classes_nums=self._class_nums)

    def call(self, input_tensor):
        """

        :param input_tensor:
        :return:
        """
        # detail branch
        detail_branch_output = self._detail_branch(input_tensor)
        # semantic branch
        semantic_branch_output, semantic_branch_seg_logits = self._semantic_branch(
            input_tensor
        )
        # build aggregation branch
        aggregation_branch_output = self._guided_aggregation_branch(
            [detail_branch_output, semantic_branch_output]
        )
        # segmentation head
        segment_logits = self._semantic_heads(
            aggregation_branch_output
        )
        semantic_branch_seg_logits['seg_head'] = segment_logits
        if self._phase == "train":
            if 1 == len(semantic_branch_seg_logits): # no boosting used
                output_tensors = tf.expand_dims(segment_logits, -1)
            else:
                output_tensors = Concatenate()([seg_head for seg_head in semantic_branch_seg_logits.values()])
            return output_tensors
        else:
            segment_score = tf.nn.softmax(logits=segment_logits, name='prob')
            segment_prediction = tf.argmax(segment_score, axis=-1, name='prediction')
            return segment_prediction

    def compute_loss(self, label_tensor, output_tensors):
        # compute network loss
        if self._loss_type == 'cross_entropy':
            if not self._enable_ohem:
                segment_loss = self._compute_cross_entropy_loss(
                    seg_logits=output_tensors,
                    labels=label_tensor,
                    class_nums=self._class_nums
                )
            else:
                segment_loss = self._compute_ohem_cross_entropy_loss(
                    seg_logits=output_tensors,
                    labels=label_tensor,
                    class_nums=self._class_nums,
                    thresh=self._ohem_score_thresh,
                    n_min=self._ohem_min_sample_nums
                )
        else:
            raise NotImplementedError('Not supported loss of type: {:s}'.format(self._loss_type))
        l2_reg_loss = self._compute_l2_reg_loss(
            var_list=self.trainable_variables,
            weights_decay=self._weights_decay,
            name='segment_l2_loss'
        )
        total_loss = segment_loss + l2_reg_loss
        return total_loss

if __name__ == '__main__':
    """
    test code
    """
    import time

    from local_utils.config_utils import parse_config_utils

    CFG = parse_config_utils.cityscapes_cfg_v2

    time_comsuming_loops = 5
    test_input = tf.random.normal(shape=[2, 512, 1024, 3], dtype=tf.float32)
    test_label = tf.random.uniform(shape=[2, 512, 1024, 1], minval=0, maxval=6, dtype=tf.int32)

    stem_block = _StemBlock(phase='train', output_channels=16)
    stem_block_output = stem_block(test_input)

    context_embedding_block = _ContextEmbedding(phase='train')
    context_embedding_block_output = context_embedding_block(
        stem_block_output
    )

    ge_output_stride_1 = _GatherExpansion(
        phase='train',
        stride=1,
        e=6,
        output_channels=128)(context_embedding_block_output)

    ge_output_stride_2 = _GatherExpansion(
        phase='train',
        stride=2,
        e=6,
        output_channels=128)(ge_output_stride_1)

    ge_output_stride_2 = _GatherExpansion(
        phase='train',
        stride=2,
        e=6,
        output_channels=128
    )(ge_output_stride_2)

    guided_aggregation_block = _GuidedAggregation(phase='train')
    guided_aggregation_block_output = _GuidedAggregation(phase='train')([stem_block_output, ge_output_stride_2])

    seg_head = _SegmentationHead(
        phase='train',
        upsample_ratio=4,
        feature_dims=64,
        classes_nums=9)
    seg_head_output = seg_head(stem_block_output)

    bisenetv2 = BiseNetKerasV2(phase="train", cfg=CFG)
    bisenetv2_detail_branch_output = _DetailBranch(phase="train")(test_input)
    bisenetv2_semantic_branch_output, segment_head_inputs = _SemanticBranch(
        phase="train",
        semantic_channel_ratio=CFG.MODEL.BISENETV2.SEMANTIC_CHANNEL_LAMBDA,
        ge_expand_ratio=CFG.MODEL.BISENETV2.GE_EXPAND_RATIO,
        seg_head_ratio=CFG.MODEL.BISENETV2.SEGHEAD_CHANNEL_EXPAND_RATIO,
        class_nums=CFG.DATASET.NUM_CLASSES)(test_input)
    bisenetv2_aggregation_output = _GuidedAggregation(phase="train")(
        [bisenetv2_detail_branch_output, bisenetv2_semantic_branch_output]
    )
    output_tensors = bisenetv2(test_input)
    loss_set = bisenetv2.compute_loss(
        label_tensor=test_label,
        output_tensors=output_tensors)
    bisenetv2._phase="predict"
    logits = bisenetv2(test_input)

    #with tf.Session() as sess:
    #    sess.run(tf.global_variables_initializer())
    #    # stem block time consuming
    #    t_start = time.time()
    #    for i in range(time_comsuming_loops):
    #        sess.run(stem_block_output)
    #    print('Stem block module cost time: {:.5f}s'.format((time.time() - t_start) / time_comsuming_loops))
    #    print(stem_block_output)

    #    # context embedding block time consuming
    #    t_start = time.time()
    #    for i in range(time_comsuming_loops):
    #        sess.run(context_embedding_block_output)
    #    print('Context embedding block module cost time: {:.5f}s'.format(
    #        (time.time() - t_start) / time_comsuming_loops)
    #    )
    #    print(context_embedding_block_output)

    #    # ge block with stride 1 time consuming
    #    t_start = time.time()
    #    for i in range(time_comsuming_loops):
    #        sess.run(ge_output_stride_1)
    #    print('Ge block with stride 1 module cost time: {:.5f}s'.format((time.time() - t_start) / time_comsuming_loops))
    #    print(ge_output_stride_1)

    #    # ge block with stride 2 time consuming
    #    t_start = time.time()
    #    for i in range(time_comsuming_loops):
    #        sess.run(ge_output_stride_2)
    #    print('Ge block with stride 2 module cost time: {:.5f}s'.format((time.time() - t_start) / time_comsuming_loops))
    #    print(ge_output_stride_2)

    #    # guided aggregation block time consuming
    #    t_start = time.time()
    #    for i in range(time_comsuming_loops):
    #        sess.run(guided_aggregation_block_output)
    #    print('Guided aggregation module cost time: {:.5f}s'.format((time.time() - t_start) / time_comsuming_loops))
    #    print(guided_aggregation_block_output)

    #    # segmentation head block time consuming
    #    t_start = time.time()
    #    for i in range(time_comsuming_loops):
    #        sess.run(seg_head_output)
    #    print('Segmentation head module cost time: {:.5f}s'.format((time.time() - t_start) / time_comsuming_loops))
    #    print(seg_head_output)

    #    # bisenetv2 detail branch time consuming
    #    t_start = time.time()
    #    for i in range(time_comsuming_loops):
    #        sess.run(bisenetv2_detail_branch_output)
    #    print('Bisenetv2 detail branch cost time: {:.5f}s'.format((time.time() - t_start) / time_comsuming_loops))
    #    print(bisenetv2_detail_branch_output)

    #    # bisenetv2 semantic branch time consuming
    #    t_start = time.time()
    #    for i in range(time_comsuming_loops):
    #        sess.run(bisenetv2_semantic_branch_output)
    #    print('Bisenetv2 semantic branch cost time: {:.5f}s'.format((time.time() - t_start) / time_comsuming_loops))
    #    print(bisenetv2_semantic_branch_output)

    #    # bisenetv2 aggregation branch time consuming
    #    t_start = time.time()
    #    for i in range(time_comsuming_loops):
    #        sess.run(bisenetv2_aggregation_output)
    #    print('Bisenetv2 aggregation branch cost time: {:.5f}s'.format((time.time() - t_start) / time_comsuming_loops))
    #    print(bisenetv2_aggregation_output)

    #    # bisenetv2 compute loss time consuming
    #    t_start = time.time()
    #    for i in range(time_comsuming_loops):
    #        sess.run(loss_set)
    #    print('Bisenetv2 compute loss cost time: {:.5f}s'.format((time.time() - t_start) / time_comsuming_loops))
    #    print(loss_set)

    #    # bisenetv2 inference time consuming
    #    t_start = time.time()
    #    for i in range(time_comsuming_loops):
    #        sess.run(logits)
    #    print('Bisenetv2 inference cost time: {:.5f}s'.format((time.time() - t_start) / time_comsuming_loops))
    #    print(logits)

