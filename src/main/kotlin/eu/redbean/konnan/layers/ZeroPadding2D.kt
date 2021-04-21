package eu.redbean.konnan.layers

import eu.redbean.kten.api.tensor.Constants.all
import eu.redbean.kten.api.tensor.Tensor

class ZeroPadding2D(
    val top: Int,
    val bottom: Int,
    val left: Int,
    val right: Int,
    val channelsFirst: Boolean = true,
    name: String? = null
): Layer(name) {

    override fun postInvoke() {
        val inShape = previousLayers[0].shape
        if (inShape.size != 3)
            throw IllegalArgumentException("ZeroPadding2D layer requires previous layer to have 3 dimensions, but got layer with shape: $inShape")

        val heightAxis = if (channelsFirst) 1 else 0
        val widthAxis = if (channelsFirst) 2 else 1

        this.shape = inShape.mapIndexed { index, size ->
            if (index == heightAxis)
                size + top + bottom
            else if (index == widthAxis)
                size + left + right
            else size
        }
    }

    override fun forward(input: Tensor): Tensor {
        val batchSize = input.shape[0]
        val res = Tensor.zeros(batchSize, *this.shape.toIntArray()).asVariable(requiresGrad = true)
        if (channelsFirst) {
            res[all, all, top until (input.shape[2] + top), left until (input.shape[3] + left)] = input
        } else {
            res[all, top until (input.shape[1] + top), left until (input.shape[2] + left), all] = input
        }
        return res
    }
}