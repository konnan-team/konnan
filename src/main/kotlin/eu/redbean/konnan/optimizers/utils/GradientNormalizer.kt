package eu.redbean.konnan.optimizers.utils

import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.tensor.Tensor
import kotlin.math.pow

fun interface GradientNormalizer {
    fun normalize(parameters: List<Variable>)
}

fun clipGradientNormalizer(maxNorm: Float, normOrder: Int = 2) = GradientNormalizer { parameters ->
    var totalNorm = 0f

    parameters.map { it.grad() }
        .forEach {
            val norm = Tensor.sum(Tensor.abs(it) pow normOrder)
            totalNorm += norm.item()
        }
    totalNorm = totalNorm.pow(1f / normOrder)

    val clipVal = maxNorm / (totalNorm + 1e-6f)

    if (clipVal < 1f) {
        parameters.forEach { it.inplaceMultiplyGrad(clipVal) }
    }
}