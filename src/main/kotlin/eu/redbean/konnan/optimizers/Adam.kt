package eu.redbean.konnan.optimizers

import eu.redbean.konnan.layers.AbstractLayerBase
import eu.redbean.konnan.optimizers.schedulers.FixedLR
import eu.redbean.konnan.optimizers.schedulers.LearningRateScheduler
import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.*
import eu.redbean.kten.api.tensor.Tensor.Companion.sqrt
import eu.redbean.kten.api.tensor.platform.PlatformProvider.epsilon
import kotlin.math.pow

class Adam(
    learningRate: Float = 0.001f,
    val beta1: Float = 0.9f,
    val beta2: Float = 0.999f,
    scheduler: LearningRateScheduler = FixedLR()
): AbstractOptimizer(learningRate, scheduler) {

    override fun updateParameter(layer: AbstractLayerBase, parameterName: String, parameterValue: Variable) {
        val momentumName = "#momentum $parameterName"
        layer.parameters.putIfAbsent(
            momentumName,
            Tensor.zerosLikeNoGrad(parameterValue)
        )
        val cacheName = "#cache $parameterName"
        layer.parameters.putIfAbsent(
            cacheName,
            Tensor.zerosLikeNoGrad(parameterValue)
        )

        layer.platform.garbageCollector().use {
            val oldMomentum = layer.parameters[momentumName]!!
            layer.parameters[momentumName] = beta1 * oldMomentum + (1.0f - beta1) * parameterValue.grad()
            val oldCache = layer.parameters[cacheName]!!
            layer.parameters[cacheName] = beta2 * oldCache + (1.0f - beta2) * (parameterValue.grad() pow 2)

            val momentumCorrected = layer.parameters[momentumName]!! / (1.0f - beta1.pow(iteration + 1))
            val cacheCorrected = layer.parameters[cacheName]!! / (1.0f - beta2.pow(iteration + 1))

            parameterValue.inplaceAddToValue(-currentLR * momentumCorrected / (sqrt(cacheCorrected) + epsilon))
            parameterValue.zeroGrad()

            it.mayRelease(oldMomentum, oldCache)
            it.mustKeep(layer.parameters[momentumName]!!, layer.parameters[cacheName]!!)
        }
    }

}