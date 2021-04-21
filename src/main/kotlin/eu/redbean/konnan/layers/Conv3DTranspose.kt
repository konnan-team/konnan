package eu.redbean.konnan.layers

import eu.redbean.konnan.layers.utils.asTriple
import eu.redbean.konnan.layers.utils.tripleOf
import eu.redbean.kten.api.autograd.functions.nn.conv3dTranspose
import eu.redbean.kten.api.tensor.Tensor

class Conv3DTranspose(
    size: Int,
    kernelSize: Triple<Int, Int, Int>,
    stride: Triple<Int, Int, Int> = tripleOf(1),
    padding: Triple<Int, Int, Int> = tripleOf(0),
    outputPadding: Triple<Int, Int, Int> = tripleOf(0),
    dilation: Triple<Int, Int, Int> = tripleOf(1),
    groups: Int = 1,
    channelsFirst: Boolean = true,
    useBias: Boolean = true,
    name: String? = null
) : AbstractConvLayer(
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

    override fun checkInputShape(inShape: List<Int>) {
        if (inShape.size != 4)
            throw IllegalArgumentException(
                "Conv3DTranspose layer expects the previous layer to have 4 dimensions: " +
                        "(C x D x H x W) in channels first mode, or (D x H x W x C) in channels last mode, but got layer with shape: ${inShape}"
            )
    }

    override fun forwardWithChannelsFirst(input: Tensor): Tensor {
        return input.conv3dTranspose(
            weight,
            if (useBias) bias else null,
            stride.asTriple(),
            padding.asTriple(),
            outputPadding.asTriple(),
            dilation.asTriple(),
            groups
        )
    }
    
}