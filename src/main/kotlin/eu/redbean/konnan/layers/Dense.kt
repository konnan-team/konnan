package eu.redbean.konnan.layers

import eu.redbean.konnan.layers.initializers.Initializer
import eu.redbean.konnan.layers.initializers.constant
import eu.redbean.konnan.layers.initializers.heNormal
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.*

class Dense(
    val size: Int,
    val useBias: Boolean = true,
    val biasInitializer: Initializer = constant(0f),
    val weightInitScale: Float = 0.1f,
    val weightInitializer: Initializer = heNormal(),
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
        weights = (weightInitScale * weightInitializer.init(previousLayers[0].shape + size))
            .toPlatform(platform.platformKey)
            .asVariable(requiresGrad = true)
        if (useBias)
            biases = biasInitializer.init(listOf(1, size)).toPlatform(platform.platformKey).asVariable(true)
    }

    override fun forward(input: Tensor): Tensor {
        if (useBias)
            return (input matmul weights) + biases
        else
            return input matmul weights
    }

}