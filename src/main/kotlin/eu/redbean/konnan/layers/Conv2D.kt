package eu.redbean.konnan.layers

import eu.redbean.konnan.layers.initializers.Initializer
import eu.redbean.konnan.layers.utils.asPair
import eu.redbean.kten.api.autograd.functions.nn.conv2d
import eu.redbean.kten.api.tensor.Tensor

class Conv2D(
    size: Int,
    kernelSize: Pair<Int, Int>,
    stride: Pair<Int, Int> = 1 to 1,
    padding: Pair<Int, Int> = 0 to 0,
    dilation: Pair<Int, Int> = 1 to 1,
    groups: Int = 1,
    channelsFirst: Boolean = true,
    useBias: Boolean = true,
    weightInitializer: Initializer? = null,
    biasInitializer: Initializer? = null,
    name: String? = null
): AbstractConvLayer(
    size,
    kernelSize.toList(),
    padding.toList(),
    dilation.toList(),
    stride.toList(),
    listOf(0, 0),
    groups,
    useBias,
    channelsFirst,
    false,
    weightInitializer,
    biasInitializer,
    name
) {

    constructor(size: Int,
                kernelSize: Int = 1,
                stride: Int = 1,
                padding: Int = 0,
                dilation: Int = 1,
                groups: Int = 1,
                channelsFirst: Boolean = true,
                useBias: Boolean = true,
                weightInitializer: Initializer? = null,
                biasInitializer: Initializer? = null,
                name: String? = null) : this(
        size,
        kernelSize to kernelSize,
        stride to stride,
        padding to padding,
        dilation to dilation,
        groups, channelsFirst, useBias, weightInitializer, biasInitializer, name
    )

    override fun checkInputShape(inShape: List<Int>) {
        if (inShape.size != 3)
            throw IllegalArgumentException("Conv2D layer expects the previous layer to have 3 dimensions: " +
                    "(C x H x W) in channels first mode, or (H x W x C) in channels last mode, but got layer with shape: ${inShape}")
    }

    override fun forwardWithChannelsFirst(input: Tensor): Tensor {
        return input.conv2d(weight, if (useBias) bias else null, stride.asPair(), padding.asPair(), dilation.asPair(), groups)
    }

}