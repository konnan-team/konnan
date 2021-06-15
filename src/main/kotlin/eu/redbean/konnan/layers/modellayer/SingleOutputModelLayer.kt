package eu.redbean.konnan.layers.modellayer

import eu.redbean.konnan.layers.AbstractLayerBase
import eu.redbean.konnan.layers.Input
import eu.redbean.konnan.layers.MultiInputLayer
import eu.redbean.konnan.layers.interfaces.HasCustomUpdate
import eu.redbean.konnan.models.Model
import eu.redbean.konnan.optimizers.AbstractOptimizer
import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.serialization.TensorSerializer

class SingleOutputModelLayer(
    private val model: Model,
    name: String? = null
) : MultiInputLayer(name), HasCustomUpdate {

    override fun postInvoke() {
        if (previousLayers.size != model.inputs.size)
            throw IllegalArgumentException("Wrong number of arguments, model layer requires ${model.inputs.size} layers as input")
        previousLayers.map(AbstractLayerBase::shape).zip(model.inputs.map(Input::shape))
            .forEachIndexed { index, (previousLayerShape, modelInputShape) ->
                if (previousLayerShape != modelInputShape)
                    throw IllegalArgumentException(
                        "Invalid argument shape, model layer got input layer at index: $index " +
                                "with shape: $previousLayerShape but the underlying model requires input at the same index " +
                                "with shape: $modelInputShape"
                    )
            }

        this.shape = model.outputs[0].shape
    }

    override fun updateParameters(optimizer: AbstractOptimizer) {
        if (trainable)
            model.updateParameters(optimizer)
    }

    override fun forward(inputs: List<Tensor>): Tensor {
        model.layers.forEach { it.training = this.training }
        model.inputs.zip(inputs).forEach { (inputLayer, input) -> inputLayer.forward(input) }
        return model.outputs[0].value
    }

    override fun <T> serializeParametersWith(serializer: TensorSerializer<T>, prefix: String): List<Pair<String, T>> {
        return model.layers.flatMapIndexed { index, layer ->
            layer.serializeParametersWith(serializer, "[$index]:")
        }
    }

    override fun <T> deserializeParametersWith(serializer: TensorSerializer<T>, serializedParametes: List<Pair<String, T>>) {
        val indexPattern = """^\[(\d+)]:(.+)$""".toRegex()

        val layerIndexedSerializedParameters = mutableMapOf<Int, MutableList<Pair<String, T>>>()

        serializedParametes.forEach {
            if (indexPattern.matches(it.first)) { //TODO error handling
                val (indexString, paramName) = indexPattern.matchEntire(it.first)!!.destructured
                val layerIndex = indexString.toInt()
                layerIndexedSerializedParameters.computeIfAbsent(layerIndex) { mutableListOf() }.add(paramName to it.second)
            }
        }

        layerIndexedSerializedParameters.forEach { layerIndex, serializedParams ->
            model.layers[layerIndex].deserializeParametersWith(serializer, serializedParams)
        }
    }

    override fun clearOutput() {
        super.clearOutput()
        model.layers.forEach { it.clearOutput() }
    }

}