package eu.redbean.konnan.layers

import eu.redbean.kten.api.autograd.utils.reshape
import eu.redbean.kten.api.tensor.Tensor

class Reshape(
    vararg val newShape: Int,
    name: String? = null
): Layer(name) {

    override fun postInvoke() {
        shape = previousLayers[0].shape.reshape(newShape.toList())
    }

    override fun forward(input: Tensor): Tensor {
        return input.reshape(-1, *shape.toIntArray()) //TODO change it when time dist layer is implemented
    }

}