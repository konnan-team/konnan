package eu.redbean.konnan.metrics

import eu.redbean.kten.api.autograd.utils.toStoreSize
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.Tensor.Companion.abs
import eu.redbean.kten.api.tensor.Tensor.Companion.mean

fun accuracy(yPred: Tensor, yTrue: Tensor, precisionScale: Float = 1.0f/250f): Float {
    val precision = yTrue.reshape(yTrue.shape.toStoreSize()).std(-1).item() * precisionScale
    return mean(abs(yPred - yTrue) lt precision).item()
}

class Accuracy(
    private val precisionScale: Float = 1.0f/250f,
    override val name: String = "accuracy"
): Metric {
    override fun calculate(yPred: Tensor, yTrue: Tensor): Float = accuracy(yPred, yTrue, precisionScale)
}

fun categoricalAccuracy(yPred: Tensor, yTrue: Tensor): Float = mean(yPred.argMax(axis = -1) eq yTrue.argMax(axis = -1)).item()

class CategoricalAccuracy(
    private val oneHotLabels: Boolean = true,
    override val name: String = "categorical_accuracy"
): Metric {
    override fun calculate(yPred: Tensor, yTrue: Tensor): Float {
        if (oneHotLabels)
            return categoricalAccuracy(yPred, yTrue)

        TODO("Convert to onehot")
    }
}

