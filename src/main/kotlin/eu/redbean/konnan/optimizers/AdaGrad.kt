package eu.redbean.konnan.optimizers

import eu.redbean.konnan.layers.AbstractLayerBase
import eu.redbean.konnan.optimizers.schedulers.FixedLR
import eu.redbean.konnan.optimizers.schedulers.LearningRateScheduler
import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.*
import eu.redbean.kten.api.tensor.Tensor.Companion.sqrt
import eu.redbean.kten.api.tensor.platform.PlatformProvider

/**
 * Adaptive Gradient Descent optimizer
 */
class AdaGrad(
    learningRate: Float,
    val epsilon: Float = PlatformProvider.epsilon,
    scheduler: LearningRateScheduler = FixedLR()
): AbstractOptimizer(learningRate, scheduler) {

    override fun updateParameter(layer: AbstractLayerBase, parameterName: String, parameterValue: Variable) {
        val parameterCacheName = "#cache $parameterName"
        layer.parameters.putIfAbsent(
            parameterCacheName,
            Tensor.zerosLikeNoGrad(parameterValue)
        )
        layer.parameters[parameterCacheName] = layer.parameters[parameterCacheName]!! + (parameterValue.grad() pow 2)
        parameterValue.inplaceAddToValue(-currentLR * parameterValue.grad() / (sqrt(layer.parameters[parameterCacheName]!!) + epsilon))
        parameterValue.zeroGrad()
    }
}