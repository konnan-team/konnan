package eu.redbean.konnan.metrics

import eu.redbean.kten.api.tensor.Tensor

fun interface Metric {
    val name: String get() = "metric"
    fun calculate(yPred: Tensor, yTrue: Tensor): Float
}