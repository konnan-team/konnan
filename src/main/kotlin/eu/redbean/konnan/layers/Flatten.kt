package eu.redbean.konnan.layers

import eu.redbean.kten.api.autograd.utils.toStoreSize
import eu.redbean.kten.api.tensor.Tensor

class Flatten(name: String? = null): Layer(name) {

    override fun postInvoke() {
        shape = listOf(previousLayers[0].shape.toStoreSize())
    }

    override fun forward(input: Tensor): Tensor {
        return input.reshape(-1, shape[0]) //TODO change it when time dist is implemented!
    }

}