package eu.redbean.konnan.optimizers

import eu.redbean.konnan.layers.AbstractLayerBase
import eu.redbean.konnan.optimizers.schedulers.FixedLR
import eu.redbean.konnan.optimizers.schedulers.LearningRateScheduler
import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.*

/**
 * Stochastic Gradient Descent optimizer
 */
class SGD(
    learningRate: Float,
    val momentum: Float = 0.0f,
    scheduler: LearningRateScheduler = FixedLR()
): AbstractOptimizer(learningRate, scheduler) {

    override fun updateParameter(layer: AbstractLayerBase, parameterName: String, parameterValue: Variable) {
        if (momentum != 0.0f) {
            val parameterMomentumName = "#momentum ${parameterName}" // TODO specify not to have parameters with names like this
            layer.parameters.putIfAbsent(
                parameterMomentumName,
                Tensor.zerosLikeNoGrad(parameterValue)
            )
            layer.platform.garbageCollector().use {
                val oldMomentum = layer.parameters[parameterMomentumName]!!

                val paramUpdates = momentum * oldMomentum - currentLR * parameterValue.grad()
                layer.parameters[parameterMomentumName] = paramUpdates
                parameterValue.inplaceAddToValue(paramUpdates)

                it.mayRelease(oldMomentum)
                it.mustKeep(layer.parameters[parameterMomentumName]!!)
            }
        } else {
            parameterValue.inplaceAddToValue(-currentLR * parameterValue.grad())
        }

        parameterValue.zeroGrad()
    }

}