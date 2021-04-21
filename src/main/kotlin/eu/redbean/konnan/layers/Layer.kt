package eu.redbean.konnan.layers

import eu.redbean.kten.api.tensor.Tensor

abstract class Layer(name: String?): AbstractLayerBase(name) {

    open operator fun invoke(inputLayer: AbstractLayerBase): AbstractLayerBase {
        previousLayers.clear()
        previousLayers.add(inputLayer)
        this.shape = inputLayer.shape.toList()
        postInvoke()
        return this
    }

    override fun doForward(): Tensor {
        if (previousLayers.size != 1) //TODO layer naming and more detailed message
            throw IllegalStateException("Non input Layer requires previous layers for forward pass, but none found")

        return forward(previousLayers[0].output!!)
    }

    abstract fun forward(input: Tensor): Tensor

}