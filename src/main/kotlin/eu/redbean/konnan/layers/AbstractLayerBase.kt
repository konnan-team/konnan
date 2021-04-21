package eu.redbean.konnan.layers

import eu.redbean.konnan.layers.naming.LayerNameProvider
import eu.redbean.kten.api.autograd.utils.toStoreSize
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.BasicTensorOperations
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.serialization.TensorSerializer

abstract class AbstractLayerBase(
    name: String?
) {

    val name = LayerNameProvider.calculateNameFor(this, name)

    open val parameters = mutableMapOf<String, Tensor>()

    var shape: List<Int> = listOf()

    val previousLayers = mutableListOf<AbstractLayerBase>()

    internal var platform: BasicTensorOperations = PlatformProvider.tensorOperations()
        set(value) {
            if (value != field) {
                previousLayers.forEach { it.platform = value }
                field = value
                platformChanged()
            }
        }

    var trainable = true
    var training = true

    internal var output: Tensor? = null

    val value: Tensor
        get() {
            if (output == null || inputsChanged())
                forwardPass()
            return output!!
        }

    open fun platformChanged() {
        parameters.keys.toList().forEach {
            parameters[it] = parameters[it]!!.toPlatform(platform.platformKey)
        }
    }

    open fun postInvoke() {
        //override if needed
    }

    fun forwardPass() {
        previousLayers.forEach(AbstractLayerBase::forwardPass)
        output = doForward()
    }

    abstract protected fun doForward(): Tensor

    open fun inputsChanged(): Boolean {
        return previousLayers.any { it.inputsChanged() }
    }

    fun summary(): List<Pair<String, Int>> {
        val previousSummary = previousLayers.flatMap { it.summary() }
        val selfSummary = "${this.name} shape: $shape" to parameters.values.fold(0) { acc, tensor ->
            acc + if (trainable) tensor.shape.toStoreSize() else 0
        }
        return previousSummary + selfSummary
    }

    open fun <T> serializeParametersWith(serializer: TensorSerializer<T>, prefix: String = ""): List<Pair<String, T>> {
        return parameters.map { prefix + it.key to it.value.serializeWith(serializer) }
    }

    open fun <T> deserializeParametersWith(serializer: TensorSerializer<T>, serializedParametes: List<Pair<String, T>>) {
        parameters.clear()
        serializedParametes.forEach {
            parameters.put(it.first, Tensor.deserializeWith(serializer, it.second).toPlatform(platform.platformKey))
        }
    }

}