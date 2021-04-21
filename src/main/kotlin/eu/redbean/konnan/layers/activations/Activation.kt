package eu.redbean.konnan.layers.activations

import eu.redbean.konnan.layers.Layer
import eu.redbean.kten.api.tensor.Tensor

open class Activation(
    private val activationFunction: (Tensor) -> Tensor,
    name: String? = null
): Layer(name) {

    override fun postInvoke() {
        this.shape = previousLayers[0].shape
    }

    override fun forward(input: Tensor): Tensor = activationFunction(input)

}