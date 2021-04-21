package eu.redbean.konnan.layers

import eu.redbean.kten.api.autograd.functions.nn.conv1d
import eu.redbean.kten.api.tensor.Tensor

class Conv1D(
    size: Int,
    kernelSize: Int = 1,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
    groups: Int = 1,
    channelsFirst: Boolean = true,
    useBias: Boolean = true,
    name: String? = null
): AbstractConvLayer(
    size,
    listOf(kernelSize),
    listOf(padding),
    listOf(dilation),
    listOf(stride),
    listOf(0),
    groups,
    useBias,
    channelsFirst,
    false,
    name
) {

    override fun checkInputShape(inShape: List<Int>) {
        if (inShape.size != 2)
            throw IllegalArgumentException("Conv1D layer expects the previous layer to have 2 dimensions: " +
                    "(C x W) in channels first mode, or (W x C) in channels last mode, but got layer with shape: ${inShape}")
    }

    override fun forwardWithChannelsFirst(input: Tensor): Tensor {
        return input.conv1d(weight, if (useBias) bias else null, stride[0], padding[0], dilation[0], groups)
    }

}