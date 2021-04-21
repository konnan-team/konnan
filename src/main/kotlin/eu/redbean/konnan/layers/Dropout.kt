package eu.redbean.konnan.layers

import eu.redbean.kten.api.tensor.Tensor

class Dropout(
    val rate: Float,
    name: String? = null
): Layer(name) {

    override fun forward(input: Tensor): Tensor {
        if (training) {
            val keepRate = 1f - rate
            return input * (platform.createBernoulli(input.shape, keepRate) / keepRate)
        }
        return input
    }

}