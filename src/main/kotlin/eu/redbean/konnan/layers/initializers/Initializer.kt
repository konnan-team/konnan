package eu.redbean.konnan.layers.initializers

import eu.redbean.kten.api.tensor.Tensor

/**
 * Parameter initializer interface. All implementation must return a tensor with the given shape,
 * the tensor is converted to variable and the platform changes also applied in the layers, so
 * there is no need to convert them in the initializer itself.
 */
fun interface Initializer {
    fun init(shape: List<Int>): Tensor
}