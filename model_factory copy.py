import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


_KERNEL_SIZE_ERROR = "kernel_size must be an integer or a pair of positive integers"
_STRIDES_ERROR = "strides must be an integer or a pair of positive integers"


@keras.utils.register_keras_serializable(package="CalcCondConv")
class GroupConv2D(layers.Layer):
    """Grouped convolution implemented as independent Conv2D operations."""

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        padding="same",
        groups=1,
        strides=1,
        kernel_initializer="glorot_uniform",
        use_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if groups < 1:
            raise ValueError("groups must be a positive integer")
        if input_channels % groups or output_channels % groups:
            raise ValueError("input_channels and output_channels must be divisible by groups")
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.strides = strides
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.use_bias = use_bias
        self.convs = [
            layers.Conv2D(
                output_channels // groups,
                kernel_size,
                padding=padding,
                strides=strides,
                kernel_initializer=keras.initializers.serialize(self.kernel_initializer),
                use_bias=use_bias,
            )
            for _ in range(groups)
        ]

    def build(self, input_shape):
        if input_shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {input_shape[-1]}"
            )
        super().build(input_shape)

    def call(self, inputs):
        splits = tf.split(inputs, num_or_size_splits=self.groups, axis=-1)
        outputs = [conv(part) for conv, part in zip(self.convs, splits)]
        return tf.concat(outputs, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_channels": self.input_channels,
                "output_channels": self.output_channels,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "groups": self.groups,
                "strides": self.strides,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "use_bias": self.use_bias,
            }
        )
        return config

@keras.utils.register_keras_serializable(package="CalcCondConv")
class CondConv2D(layers.Layer):
    """Input-conditioned convolution assembled from a bank of expert kernels."""

    def __init__(
        self,
        filters,
        kernel_size,
        num_experts=4,
        strides=1,
        padding="same",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if filters < 1 or num_experts < 1:
            raise ValueError("filters and num_experts must be positive")
        self.filters = filters
        self.kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        if len(self.kernel_size) != 2 or any(size < 1 for size in self.kernel_size):
            raise ValueError(_KERNEL_SIZE_ERROR)
        self.num_experts = num_experts
        self.strides = (strides, strides) if isinstance(strides, int) else tuple(strides)
        if len(self.strides) != 2 or any(stride < 1 for stride in self.strides):
            raise ValueError(_STRIDES_ERROR)
        self.padding = padding.upper()
        if self.padding not in {"SAME", "VALID"}:
            raise ValueError("padding must be 'same' or 'valid'")
        self.expert_kernels = None
        self.router = None

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if input_channels is None:
            raise ValueError("CondConv2D requires a known input channel dimension")
        self.expert_kernels = self.add_weight(
            name="expert_kernels",
            shape=(
                self.num_experts,
                self.kernel_size[0],
                self.kernel_size[1],
                input_channels,
                self.filters,
            ),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.router = layers.Dense(self.num_experts, activation="sigmoid")
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        input_channels = tf.shape(inputs)[-1]
        routing = self.router(tf.reduce_mean(inputs, axis=[1, 2]))
        dynamic_kernel = tf.einsum(
            "bk,kxyio->bxyio", routing, self.expert_kernels
        )

        packed_inputs = tf.reshape(
            tf.transpose(inputs, [1, 2, 0, 3]),
            [1, tf.shape(inputs)[1], tf.shape(inputs)[2], batch_size * input_channels],
        )
        packed_kernel = tf.reshape(
            tf.transpose(dynamic_kernel, [1, 2, 3, 0, 4]),
            [
                self.kernel_size[0],
                self.kernel_size[1],
                input_channels,
                batch_size * self.filters,
            ],
        )
        outputs = tf.nn.conv2d(
            packed_inputs,
            packed_kernel,
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding,
        )
        outputs = tf.reshape(
            outputs,
            [
                tf.shape(outputs)[1],
                tf.shape(outputs)[2],
                batch_size,
                self.filters,
            ],
        )
        outputs = tf.transpose(outputs, [2, 0, 1, 3])
        outputs.set_shape(
            (inputs.shape[0], None, None, self.filters)
        )
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "num_experts": self.num_experts,
                "strides": self.strides,
                "padding": self.padding.lower(),
            }
        )
        return config


@keras.utils.register_keras_serializable(package="CalcCondConv")
class BasisConv2D(layers.Layer):
    """Sum grouped depthwise-basis responses, each followed by a channel mixer."""

    def __init__(self, channels, basis_factor=1, groups=1, strides=1, **kwargs):
        super().__init__(**kwargs)
        if channels < 1 or basis_factor < 1 or groups < 1:
            raise ValueError("channels, basis_factor, and groups must be positive")
        if basis_factor != groups:
            raise ValueError(
                "basis_factor must equal groups so each group has one spatial basis per input channel"
            )
        self.channels = channels
        self.basis_factor = basis_factor
        self.groups = groups
        self.basis_channels = channels * basis_factor
        self.strides = (strides, strides) if isinstance(strides, int) else tuple(strides)
        if len(self.strides) != 2 or any(stride < 1 for stride in self.strides):
            raise ValueError(_STRIDES_ERROR)

    def build(self, input_shape):
        if input_shape[-1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} input channels, got {input_shape[-1]}"
            )
        self.basis = self.add_weight(
            name="basis",
            shape=(3, 3, self.basis_channels),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.mix = self.add_weight(
            name="mix",
            shape=(self.groups, self.channels, self.channels),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.channels,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        outputs = None
        for group_index in range(self.groups):
            start = group_index * self.channels
            stop = start + self.channels
            depthwise_kernel = self.basis[:, :, start:stop, tf.newaxis]
            spatial = tf.nn.depthwise_conv2d(
                inputs,
                depthwise_kernel,
                strides=[1, self.strides[0], self.strides[1], 1],
                padding="SAME",
            )
            group_output = tf.einsum("bhwi,io->bhwo", spatial, self.mix[group_index])
            outputs = group_output if outputs is None else outputs + group_output
        outputs = tf.nn.bias_add(outputs, self.bias)
        outputs.set_shape((inputs.shape[0], None, None, self.channels))
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "basis_factor": self.basis_factor,
                "groups": self.groups,
                "strides": self.strides,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="CalcCondConv")
class DynamicMixerConv2D(layers.Layer):
    """Dynamic spatial convolution with a diagonal-plus-low-rank channel mixer."""

    def __init__(self, channels, reduction=4, rank=8, strides=1, **kwargs):
        super().__init__(**kwargs)
        if channels < 1 or reduction < 1 or rank < 1:
            raise ValueError("channels, reduction, and rank must be positive")
        self.channels = channels
        self.reduction = reduction
        self.mixer_hidden = max(channels // reduction, 1)
        self.rank = rank
        self.strides = (strides, strides) if isinstance(strides, int) else tuple(strides)
        if len(self.strides) != 2 or any(stride < 1 for stride in self.strides):
            raise ValueError(_STRIDES_ERROR)
        self.mixer_generator = keras.Sequential(
            [
                layers.GlobalAveragePooling2D(),
                layers.Dense(self.mixer_hidden, activation="relu"),
                layers.Dense(channels + 2 * channels * rank),
            ]
        )

    def build(self, input_shape):
        if input_shape[-1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} input channels, got {input_shape[-1]}"
            )
        self.basis = self.add_weight(
            name="basis",
            shape=(3, 3, self.channels),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.channels,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        channels = self.channels
        rank = self.rank

        params = self.mixer_generator(inputs)
        diagonal = 1.0 + 0.1 * tf.tanh(params[:, :channels])
        a = tf.reshape(
            params[:, channels : channels + channels * rank],
            [batch_size, channels, rank],
        )
        a = 0.1 * tf.tanh(a) / tf.sqrt(tf.cast(rank, a.dtype))
        b = tf.reshape(
            params[:, channels + channels * rank :],
            [batch_size, channels, rank],
        )
        b = tf.nn.softmax(b, axis=-1)

        # Filter each channel spatially first. Since the generated channel mixer
        # is constant over a sample's spatial positions, this is equivalent to
        # mixing each spatial basis kernel into a full per-sample Conv2D kernel.
        spatial = tf.nn.depthwise_conv2d(
            inputs,
            tf.cast(self.basis[:, :, :, tf.newaxis], inputs.dtype),
            strides=[1, self.strides[0], self.strides[1], 1],
            padding="SAME",
        )

        diagonal_output = spatial * tf.cast(diagonal[:, tf.newaxis, tf.newaxis, :], spatial.dtype)
        low_rank = tf.einsum(
            "bhwc,bcr->bhwr", spatial, tf.cast(a, spatial.dtype)
        )
        low_rank_output = tf.einsum(
            "bhwr,bcr->bhwc", low_rank, tf.cast(b, spatial.dtype)
        )
        outputs = diagonal_output + low_rank_output
        outputs = tf.nn.bias_add(outputs, tf.cast(self.bias, outputs.dtype))
        outputs.set_shape((inputs.shape[0], None, None, self.channels))
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "reduction": self.reduction,
                "rank": self.rank,
                "strides": self.strides,
            }
        )
        return config


class _DynamicDepthwiseBase(layers.Layer):
    def __init__(
        self,
        channels,
        num_bases=32,
        reduction=4,
        strides=1,
        coefficient_rank=8,
        kernel_size=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if channels < 1 or num_bases < 1 or reduction < 1 or coefficient_rank < 1:
            raise ValueError(
                "channels, num_bases, reduction, and coefficient_rank must be positive"
            )
        self.channels = channels
        self.num_bases = num_bases
        self.reduction = reduction
        self.coefficient_rank = coefficient_rank
        self.kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        if len(self.kernel_size) != 2 or any(size < 1 for size in self.kernel_size):
            raise ValueError(_KERNEL_SIZE_ERROR)
        self.strides = (strides, strides) if isinstance(strides, int) else tuple(strides)
        self.gap = layers.GlobalAveragePooling2D()
        hidden = max(channels // reduction, 8)
        self.mlp = keras.Sequential(
            [
                layers.Dense(hidden, activation="relu"),
                layers.Dense(self._coefficient_count(), activation=None),
            ]
        )

    def _coefficient_count(self):
        raise NotImplementedError

    def build(self, input_shape):
        if input_shape[-1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} input channels, got {input_shape[-1]}"
            )
        super().build(input_shape)

    def _apply_depthwise_kernel(self, inputs, kernel):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        kernel_height, kernel_width = self.kernel_size
        packed_inputs = tf.reshape(
            tf.transpose(inputs, [1, 2, 0, 3]),
            [1, height, width, batch_size * self.channels],
        )
        packed_kernel = tf.reshape(
            tf.transpose(kernel, [1, 2, 0, 3]),
            [kernel_height, kernel_width, batch_size * self.channels, 1],
        )
        outputs = tf.nn.depthwise_conv2d(
            packed_inputs,
            packed_kernel,
            strides=[1, self.strides[0], self.strides[1], 1],
            padding="SAME",
        )
        outputs = tf.reshape(
            outputs,
            [
                tf.shape(outputs)[1],
                tf.shape(outputs)[2],
                batch_size,
                self.channels,
            ],
        )
        outputs = tf.transpose(outputs, [2, 0, 1, 3])
        outputs.set_shape((inputs.shape[0], None, None, self.channels))
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "num_bases": self.num_bases,
                "reduction": self.reduction,
                "strides": self.strides,
                "coefficient_rank": self.coefficient_rank,
                "kernel_size": self.kernel_size,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="CalcCondConv")
class DynamicDepthwiseConv(_DynamicDepthwiseBase):
    """Mixes per-channel spatial kernels from a learned basis per input image."""

    def _coefficient_count(self):
        return self.num_bases

    def build(self, input_shape):
        super().build(input_shape)
        self.basis = self.add_weight(
            name="basis",
            shape=(
                self.kernel_size[0],
                self.kernel_size[1],
                self.num_bases,
                self.channels,
            ),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        coefficients = tf.nn.softmax(self.mlp(self.gap(inputs)), axis=-1)
        kernel = tf.einsum("bk,ijkc->bijc", coefficients, self.basis)
        return self._apply_depthwise_kernel(inputs, kernel)


@keras.utils.register_keras_serializable(package="CalcCondConv")
class DynamicBasisDepthwiseConv(_DynamicDepthwiseBase):
    """Mixes shared spatial kernels with per-image, per-channel coefficients."""

    def _coefficient_count(self):
        return self.num_bases * self.channels

    def build(self, input_shape):
        super().build(input_shape)
        self.basis = self.add_weight(
            name="basis",
            shape=(self.kernel_size[0], self.kernel_size[1], self.num_bases),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        mixing = self.mlp(self.gap(inputs))
        mixing = tf.reshape(mixing, [batch_size, self.num_bases, self.channels])
        mixing = tf.nn.softmax(mixing, axis=1)
        kernel = tf.einsum("ijk,bkc->bijc", self.basis, mixing)
        return self._apply_depthwise_kernel(inputs, kernel)


@keras.utils.register_keras_serializable(package="CalcCondConv")
class LowRankDynamicBasisDepthwiseConv(_DynamicDepthwiseBase):
    """Dynamic basis depthwise convolution with factorized channel mixing."""

    def _coefficient_count(self):
        return self.num_bases * self.coefficient_rank

    def build(self, input_shape):
        super().build(input_shape)
        self.basis = self.add_weight(
            name="basis",
            shape=(self.kernel_size[0], self.kernel_size[1], self.num_bases),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.channel_factors = self.add_weight(
            name="channel_factors",
            shape=(self.coefficient_rank, self.channels),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        latent = self.mlp(self.gap(inputs))
        latent = tf.reshape(
            latent,
            [batch_size, self.num_bases, self.coefficient_rank],
        )
        mixing = tf.einsum("bkr,rc->bkc", latent, self.channel_factors)
        mixing = tf.nn.softmax(mixing, axis=1)
        kernel = tf.einsum("ijk,bkc->bijc", self.basis, mixing)
        return self._apply_depthwise_kernel(inputs, kernel)


@keras.utils.register_keras_serializable(package="CalcCondConv")
class ResidualBlock(layers.Layer):
    """Basic residual block with a configurable 3x3 convolution operator."""

    def __init__(
        self,
        filters,
        stride=1,
        conv_type="standard",
        groups=1,
        num_bases=32,
        reduction=4,
        coefficient_rank=8,
        kernel_size=3,
        num_experts=4,
        basis_factor=1,
        basis_groups=1,
        mixer_reduction=4,
        mixer_rank=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        valid_types = {
            "standard",
            "grouped",
            "depthwise_separable",
            "condconv",
            "basis_conv",
            "dynamic_mixer",
            "dynamic_depthwise",
            "dynamic_basis_depthwise",
            "dynamic_basis_depthwise_low_rank",
        }
        if conv_type not in valid_types:
            raise ValueError(f"conv_type must be one of {sorted(valid_types)}")
        if groups < 1:
            raise ValueError("groups must be a positive integer")
        if conv_type == "grouped" and groups < 2:
            raise ValueError("groups must be at least 2 for grouped convolution")
        self.filters = filters
        self.stride = stride
        self.conv_type = conv_type
        self.groups = groups
        self.num_bases = num_bases
        self.reduction = reduction
        self.coefficient_rank = coefficient_rank
        self.kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        self.num_experts = num_experts
        self.basis_factor = basis_factor
        self.basis_groups = basis_groups
        self.mixer_reduction = mixer_reduction
        self.mixer_rank = mixer_rank
        self.conv1 = None
        self.bn1 = layers.BatchNormalization()
        self.conv2 = None
        self.bn2 = layers.BatchNormalization()
        self.projection = None

    def _make_conv(self, input_channels, output_channels, stride):
        if self.conv_type == "standard":
            return layers.Conv2D(
                output_channels,
                3,
                strides=stride,
                padding="same",
                use_bias=False,
            )
        if self.conv_type == "grouped":
            return GroupConv2D(
                input_channels,
                output_channels,
                groups=self.groups,
                strides=stride,
                use_bias=False,
            )
        if self.conv_type == "depthwise_separable":
            return keras.Sequential(
                [
                    layers.DepthwiseConv2D(
                        3,
                        strides=stride,
                        padding="same",
                        use_bias=False,
                    ),
                    layers.Conv2D(output_channels, 1, use_bias=False),
                ]
            )
        if self.conv_type == "condconv":
            return CondConv2D(
                output_channels,
                kernel_size=self.kernel_size,
                num_experts=self.num_experts,
                strides=stride,
                padding="same",
            )
        if self.conv_type == "basis_conv":
            basis_conv = BasisConv2D(
                input_channels,
                basis_factor=self.basis_factor,
                groups=self.basis_groups,
                strides=stride,
            )
            if input_channels == output_channels:
                return basis_conv
            return keras.Sequential(
                [basis_conv, layers.Conv2D(output_channels, 1, use_bias=False)]
            )
        if self.conv_type == "dynamic_mixer":
            dynamic_mixer = DynamicMixerConv2D(
                input_channels,
                reduction=self.mixer_reduction,
                rank=self.mixer_rank,
                strides=stride,
            )
            if input_channels == output_channels:
                return dynamic_mixer
            return keras.Sequential(
                [dynamic_mixer, layers.Conv2D(output_channels, 1, use_bias=False)]
            )

        dynamic_layers = {
            "dynamic_depthwise": DynamicDepthwiseConv,
            "dynamic_basis_depthwise": DynamicBasisDepthwiseConv,
            "dynamic_basis_depthwise_low_rank": LowRankDynamicBasisDepthwiseConv,
        }
        dynamic_layer = dynamic_layers[self.conv_type]
        return keras.Sequential(
            [
                dynamic_layer(
                    input_channels,
                    num_bases=self.num_bases,
                    reduction=self.reduction,
                    coefficient_rank=self.coefficient_rank,
                    kernel_size=self.kernel_size,
                    strides=stride,
                ),
                layers.Conv2D(output_channels, 1, use_bias=False),
            ]
        )

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if input_channels is None:
            raise ValueError("ResidualBlock requires a known input channel dimension")
        self.conv1 = self._make_conv(input_channels, self.filters, self.stride)
        self.conv2 = self._make_conv(self.filters, self.filters, 1)

        if self.stride != 1 or input_channels != self.filters:
            self.projection = keras.Sequential(
                [
                    layers.Conv2D(
                        self.filters,
                        1,
                        strides=self.stride,
                        padding="same",
                        use_bias=False,
                    ),
                    layers.BatchNormalization(),
                ]
            )
        super().build(input_shape)

    def call(self, inputs, training=None):
        shortcut = (
            inputs
            if self.projection is None
            else self.projection(inputs, training=training)
        )
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        return tf.nn.relu(x + shortcut)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "stride": self.stride,
                "conv_type": self.conv_type,
                "groups": self.groups,
                "num_bases": self.num_bases,
                "reduction": self.reduction,
                "coefficient_rank": self.coefficient_rank,
                "kernel_size": self.kernel_size,
                "num_experts": self.num_experts,
                "basis_factor": self.basis_factor,
                "basis_groups": self.basis_groups,
                "mixer_reduction": self.mixer_reduction,
                "mixer_rank": self.mixer_rank,
            }
        )
        return config


def build_resnet(config):
    """Build a ResNet using the selected residual-block convolution operator."""
    inputs = keras.Input(shape=(config.image_size, config.image_size, 3))
    model_config = config.model
    conv_type = model_config.get("conv_type", "standard")
    groups = model_config.get("groups", 2)
    num_bases = model_config.get("num_bases", 32)
    reduction = model_config.get("reduction", 4)
    coefficient_rank = model_config.get("coefficient_rank", 8)
    kernel_size = model_config.get("kernel_size", 3)
    num_experts = model_config.get("num_experts", 4)
    basis_factor = model_config.get("basis_factor", 1)
    basis_groups = model_config.get("basis_groups", 1)
    mixer_reduction = model_config.get("mixer_reduction", 4)
    mixer_rank = model_config.get("mixer_rank", 8)
    stem_channels = model_config["stem_channels"]
    stage_channels = model_config["stage_channels"]
    stage_depths = model_config["stage_depths"]

    x = layers.Conv2D(
        stem_channels, 5, strides=2, padding="same", use_bias=False
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for stage_index, (channels, depth) in enumerate(
        zip(stage_channels, stage_depths)
    ):
        for block_index in range(depth):
            stride = 2 if stage_index > 0 and block_index == 0 else 1
            x = ResidualBlock(
                channels,
                stride=stride,
                conv_type=conv_type,
                groups=groups,
                num_bases=num_bases,
                reduction=reduction,
                coefficient_rank=coefficient_rank,
                kernel_size=kernel_size,
                num_experts=num_experts,
                basis_factor=basis_factor,
                basis_groups=basis_groups,
                mixer_reduction=mixer_reduction,
                mixer_rank=mixer_rank,
            )(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(model_config["dropout_rate"])(x)
    outputs = layers.Dense(config.num_classes)(x)
    return keras.Model(inputs, outputs, name=f"resnet_{conv_type}")
