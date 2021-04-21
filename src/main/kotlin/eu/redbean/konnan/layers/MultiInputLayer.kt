package eu.redbean.konnan.layers

import eu.redbean.konnan.optimizers.AbstractOptimizer
import eu.redbean.kten.api.tensor.Tensor

abstract class MultiInputLayer(name: String?): AbstractLayerBase(name) {

    operator fun invoke(vararg inputLayers: AbstractLayerBase): AbstractLayerBase {
        previousLayers.clear()
        previousLayers.addAll(inputLayers)
        postInvoke()
        return this
    }

    override fun doForward(): Tensor {
        if (previousLayers.isEmpty())
            throw IllegalStateException("Non input MultiInputLayer requires previous layers, but none found")

        return forward(previousLayers.map { it.output!! })
    }

    abstract fun forward(inputs: List<Tensor>): Tensor

}