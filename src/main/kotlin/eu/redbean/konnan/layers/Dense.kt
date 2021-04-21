package eu.redbean.konnan.layers

import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.*

class Dense(
    val size: Int,
    val useBias: Boolean = true,
    val biasInitializer: (Int) -> Float = { 0f }, //TODO configurable initializers
    name: String? = null
): Layer(name) {

    var weights: Tensor by parameters
    var biases: Tensor by parameters

    init {
        if (size < 1)
            throw IllegalArgumentException("Dense layer size must be at least 1")
    }

    override fun postInvoke() {
        shape = previousLayers[0].shape.dropLast(1) + size
        weights = (0.1f * platform.createRandom(previousLayers[0].shape + size)).asVariable(requiresGrad = true)
        if (useBias)
            biases = platform.create(listOf(1, size), true, biasInitializer)
    }

    override fun forward(input: Tensor): Tensor {
        if (useBias)
            return (input matmul weights) + biases
        else
            return input matmul weights
    }

}