package eu.redbean.konnan.layers

import eu.redbean.kten.api.tensor.Tensor

class Input(
    vararg inputShape: Int,
    name: String? = null
): Layer(name) {

    private var inputsChanged = false

    init {
        this.shape = inputShape.toList()
    }

    override fun forward(input: Tensor): Tensor {
        if (input.shape.drop(1) != this.shape)
            throw IllegalArgumentException("Illegal input value. " +
                    "Input layer with shape: ${listOf("<batch>") + this.shape} got input value with shape: ${input.shape}")

        this.output = input
        inputsChanged = true
        return input
    }

    override fun doForward(): Tensor {
        inputsChanged = false
        return this.output!!
    }

    override fun invoke(inputLayer: AbstractLayerBase): AbstractLayerBase {
        throw IllegalArgumentException("Input layers do not accept layers as input")
    }

    override fun inputsChanged(): Boolean {
        return inputsChanged
    }
}