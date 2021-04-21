package eu.redbean.konnan.layers

import eu.redbean.kten.api.tensor.Tensor

class Lambda(
    outputShape: List<Int>,
    private val expression: (inputs: List<Tensor>) -> Tensor,
    name: String? = null
): MultiInputLayer(name) {

    init {
        this.shape = outputShape
    }

    override fun forward(inputs: List<Tensor>): Tensor {
        val res = expression(inputs)
        if (res.requiresGrad.not()) {
            throw IllegalStateException("Invalid Lambda layer: output must require gradients.")
        }
        val outputShapeWithoutBatch = res.shape.drop(1) //TODO change it when time dist is implemented!
        if (outputShapeWithoutBatch != this.shape) {
            throw IllegalStateException("Invalid Lambda layer: actual output shape: ${outputShapeWithoutBatch} doesn't match " +
                    "the specified output shape: ${this.shape}")
        }
        return res
    }
}