package eu.redbean.konnan.lossfunctions

import eu.redbean.kten.api.tensor.Tensor

fun interface LossFunction {
    fun calculateLoss(yPred: Tensor, yTrue: Tensor): Tensor
}