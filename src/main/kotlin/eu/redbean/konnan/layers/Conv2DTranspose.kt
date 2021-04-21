package eu.redbean.konnan.layers

import eu.redbean.konnan.layers.utils.asPair
import eu.redbean.kten.api.autograd.functions.nn.conv2dTranspose
import eu.redbean.kten.api.tensor.Tensor

class Conv2DTranspose(
    size: Int,
    kernelSize: Pair<Int, Int>,
    stride: Pair<Int, Int> = 1 to 1,
    padding: Pair<Int, Int> = 0 to 0,
    outputPadding: Pair<Int, Int> = 0 to 0,
    dilation: Pair<Int, Int> = 1 to 1,
    groups: Int = 1,
    channelsFirst: Boolean = true,
    useBias: Boolean = true,
    name: String? = null
): AbstractConvLayer(
    size,
    kernelSize.toList(),
    padding.toList(),
    dilation.toList(),
    stride.toList(),
    outputPadding.toList(),
    groups,
    useBias,
    channelsFirst,
    true,
    name
) {

    constructor(size: Int,
                kernelSize: Int = 1,
                stride: Int = 1,
                padding: Int = 0,
                outputPadding: Int = 0,
                dilation: Int = 1,
                groups: Int = 1,
                channelsFirst: Boolean = true,
                useBias: Boolean = true,
                name: String? = null) : this(
        size,
        kernelSize to kernelSize,
        stride to stride,
        padding to padding,
        outputPadding to outputPadding,
        dilation to dilation,
        groups, channelsFirst, useBias, name
    )

    override fun checkInputShape(inShape: List<Int>) {
        if (inShape.size != 3)
            throw IllegalArgumentException("Conv2DTranspose layer expects the previous layer to have 3 dimensions: " +
                    "(C x H x W) in channels first mode, or (H x W x C) in channels last mode, but got layer with shape: ${inShape}")
    }

    override fun forwardWithChannelsFirst(input: Tensor): Tensor {
        return input.conv2dTranspose(weight, if (useBias) bias else null, stride.asPair(), padding.asPair(), outputPadding.asPair(), dilation.asPair(), groups)
    }

}