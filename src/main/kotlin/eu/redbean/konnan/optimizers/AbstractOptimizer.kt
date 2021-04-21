package eu.redbean.konnan.optimizers

import eu.redbean.konnan.layers.AbstractLayerBase
import eu.redbean.konnan.optimizers.schedulers.FixedLR
import eu.redbean.konnan.optimizers.schedulers.HasSerializableState
import eu.redbean.konnan.optimizers.schedulers.LearningRateScheduler
import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.tensor.Tensor

abstract class AbstractOptimizer(
    val learningRate: Float,
    val scheduler: LearningRateScheduler = FixedLR()
) {

    var currentLR: Float = learningRate
        get private set

    protected var iteration: Int = 0

    open fun preUpdate(currentLoss: Float) {
        currentLR = scheduler.getLearningRate(iteration, learningRate, currentLR, currentLoss)
    }

    fun updateParameters(layer: AbstractLayerBase) {
        if (layer.trainable.not() || layer.parameters.isEmpty())
            return

        layer.parameters
            .filterKeys { !it.startsWith("#") }
            .map { checkParameterValidity(layer, it.key, it.value) }
            .forEach(this::updateParameter::spread)
    }

    protected abstract fun updateParameter(layer: AbstractLayerBase, parameterName: String, parameterValue: Variable)

    protected fun checkParameterValidity(layer: AbstractLayerBase, parameterName: String, parameterValue: Tensor): Triple<AbstractLayerBase, String, Variable> {
        if (parameterValue !is Variable)
            throw IllegalStateException("Invalid parameter in layer: $layer Parameter: $parameterName is not a Variable. " +
                    "Only Variables (explicitly instantiated Tensors requiring gradients) are allowed as layer parameters.")

        return Triple(layer, parameterName, parameterValue)
    }

    open fun postUpdate() {
        iteration++
    }

    fun createSerializableState(): Map<String, Number> {
        val res = mutableMapOf<String, Number>()
        res += "currentLR" to currentLR
        res += "iteration" to iteration
        if (scheduler is HasSerializableState)
            res.putAll(scheduler.getState())
        return res
    }

    fun setStateFromStateMap(state: Map<String, Number>) {
        currentLR = state["currentLR"]?.toFloat()?:currentLR
        iteration = state["iteration"]?.toInt()?:iteration
        if (scheduler is HasSerializableState)
            scheduler.setState(state)
    }

}

fun <A, B, C>((A, B, C) -> Unit).spread(t: Triple<A, B, C>) = invoke(t.first, t.second, t.third)
