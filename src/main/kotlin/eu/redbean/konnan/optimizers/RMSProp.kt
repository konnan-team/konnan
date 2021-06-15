package eu.redbean.konnan.optimizers

import eu.redbean.konnan.layers.AbstractLayerBase
import eu.redbean.konnan.optimizers.schedulers.FixedLR
import eu.redbean.konnan.optimizers.schedulers.LearningRateScheduler
import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.times

/**
 * Root Mean Square Propagation optimizer
 */
class RMSProp(
    learningRate: Float = 0.001f,
    val rho: Float = 0.9f,
    val epsilon: Float = PlatformProvider.epsilon,
    scheduler: LearningRateScheduler = FixedLR()
): AbstractOptimizer(learningRate, scheduler) {

    override fun updateParameter(layer: AbstractLayerBase, parameterName: String, parameterValue: Variable) {
        val parameterCacheName = "#cache $parameterName"
        layer.parameters.putIfAbsent(
            parameterCacheName,
            Tensor.zerosLikeNoGrad(parameterValue)
        )
        layer.platform.garbageCollector().use {
            val oldCache = layer.parameters[parameterCacheName]!!

            layer.parameters[parameterCacheName] = rho * oldCache + (1.0f - rho) * (parameterValue.grad() pow 2)
            parameterValue.inplaceAddToValue(-currentLR * parameterValue.grad() / (Tensor.sqrt(layer.parameters[parameterCacheName]!!) + epsilon))
            parameterValue.zeroGrad()

            it.mayRelease(oldCache)
            it.mustKeep(layer.parameters[parameterCacheName]!!)
        }
    }
}