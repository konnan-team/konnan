package eu.redbean.konnan.lossfunctions

import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.Tensor.Companion.abs
import eu.redbean.kten.api.tensor.Tensor.Companion.log
import eu.redbean.kten.api.tensor.Tensor.Companion.mean
import eu.redbean.kten.api.tensor.*
import eu.redbean.kten.api.tensor.platform.PlatformProvider

fun mae(yPred: Tensor, yTrue: Tensor): Tensor = mean(abs(yPred - yTrue))

class MeanAbsoluteError: LossFunction {
    override fun calculateLoss(yPred: Tensor, yTrue: Tensor): Tensor = mae(yPred, yTrue)
}

fun mse(yPred: Tensor, yTrue: Tensor): Tensor = mean((yTrue - yPred) pow 2)

class MeanSquaredError: LossFunction {
    override fun calculateLoss(yPred: Tensor, yTrue: Tensor): Tensor = mse(yPred, yTrue)
}

fun crossEntropy(yPred: Tensor, yTrue: Tensor): Tensor {
    val yPredClipped = (yPred as AGTensor).gradientAggregate {
        (it / it.sum(axis = -1, keepDimensions = true)).clamp(PlatformProvider.epsilon, 1.0f - PlatformProvider.epsilon)
    }
    return mean(-(yTrue * log(yPredClipped)).sum(axis = -1))
}

class CategoricalCrossEntropy(
    private val oneHotLabels: Boolean = true
): LossFunction {
    override fun calculateLoss(yPred: Tensor, yTrue: Tensor): Tensor {
        if (oneHotLabels)
            return crossEntropy(yPred, yTrue)

        TODO("Convert to onehot first")
    }
}

fun binaryCrossEntropy(yPred: Tensor, yTrue: Tensor): Tensor {
    val yPredClipped = yPred.clamp(PlatformProvider.epsilon, 1.0f - PlatformProvider.epsilon)
    return mean(-1.0f * (yTrue * log(yPredClipped) + (1.0f - yTrue) * log(1.0f - yPredClipped)))
}

class BinaryCrossEntropy: LossFunction {
    override fun calculateLoss(yPred: Tensor, yTrue: Tensor): Tensor = binaryCrossEntropy(yPred, yTrue)
}

fun klDiv(yPred: Tensor, yTrue: Tensor): Tensor {
    val yPredClipped = yPred.clamp(PlatformProvider.epsilon, 1.0f)
    val yTrueClipped = yTrue.clamp(PlatformProvider.epsilon, 1.0f)
    return mean(yTrueClipped * log(yTrueClipped / yPredClipped))
}

class KullbackLeiblerDivergence: LossFunction {
    override fun calculateLoss(yPred: Tensor, yTrue: Tensor): Tensor = klDiv(yPred, yTrue)
}

fun hinge(yPred: Tensor, yTrue: Tensor): Tensor = mean((1.0f - yTrue * yPred).clamp(min = 0.0f))

class Hinge: LossFunction {
    override fun calculateLoss(yPred: Tensor, yTrue: Tensor): Tensor = hinge(yPred, yTrue)
}

/**
 * 0.5 * (yPred - yTrue)^2              if |yPred - yTrue| <= delta
 * delta * |yPred - yTrue| * delta^2    if |yPred - yTrue| > delta
 */
fun huber(yPred: Tensor, yTrue: Tensor, delta: Float = 1.0f): Tensor {
    val error = yPred - yTrue
    val absError = abs(error)

    val absErrorNoGrad = absError.noGrad()
    val mask = absErrorNoGrad lte delta
    val inverseMask = absErrorNoGrad gt delta

    return mean(mask * 0.5f * (error pow 2) + inverseMask * (delta * absError - 0.5f * delta * delta))
} // TODO test (not sure if the gradients are correct with this implementation)

class Huber(
    private val delta: Float = 1.0f
): LossFunction {
    override fun calculateLoss(yPred: Tensor, yTrue: Tensor): Tensor = huber(yPred, yTrue, delta)
}

fun poisson(yPred: Tensor, yTrue: Tensor): Tensor = mean(yPred - yTrue * log(yPred + PlatformProvider.epsilon))

class Poisson: LossFunction {
    override fun calculateLoss(yPred: Tensor, yTrue: Tensor): Tensor = poisson(yPred, yTrue)
}


