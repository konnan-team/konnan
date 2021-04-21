package eu.redbean.konnan.layers

import eu.redbean.kten.api.autograd.utils.concatShapes
import eu.redbean.kten.api.autograd.utils.normalizeAxis
import eu.redbean.kten.api.tensor.Tensor

class Concatenate(
    private val axis: Int = 0,
    name: String? = null
): MultiInputLayer(name) {

    private var normalizedAxis = axis

    override fun postInvoke() {
        val shapes = previousLayers.map { it.shape }
        this.shape = concatShapes(shapes, axis)
        this.normalizedAxis = shape.normalizeAxis(axis)
    }

    override fun forward(inputs: List<Tensor>): Tensor {
        return Tensor.concat(inputs, normalizedAxis + 1)
    }


}