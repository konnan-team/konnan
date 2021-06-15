package eu.redbean.konnan.layers.activations

import eu.redbean.konnan.layers.initializers.Initializer
import eu.redbean.konnan.layers.initializers.constant
import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.plus
import eu.redbean.kten.api.tensor.times
import kotlin.math.PI
import kotlin.math.sqrt

class ReLU : Activation({ it.clamp(min = 0.0f) })

class LeakyReLU(
    negativeSlope: Float = 1e-2f
) : Activation({
    val itNG = it.noGrad()
    val mask = (itNG gte 0f) + (itNG lt 0f) * negativeSlope
    it * mask
})

class Sigmoid : Activation({ Tensor.sigmoid(it) })

class Tanh : Activation({ Tensor.tanh(it) })

class Linear : Activation({ it })

abstract class GradAggregateActivation(
    activationFunction: (Tensor) -> Tensor,
    name: String? = null
) : Activation({ (it as AGTensor).gradientAggregate(activationFunction) }, name)

class Softmax : GradAggregateActivation({
    val expVal = Tensor.exp(it - it.max(axis = -1, keepDimensions = true))
    expVal / expVal.sum(axis = -1, keepDimensions = true)
})

private fun elu(alpha: Float, input: Tensor): Tensor {
    return input.clamp(min = 0.0f) + (alpha * (Tensor.exp(input) - 1f)).clamp(max = 0.0f)
}

class ELU(alpha: Float = 1.0f) : GradAggregateActivation({ elu(alpha, it) })

/**
 * More details in Self-Normalizing Neural Networks ( https://arxiv.org/abs/1706.02515 )
 */
class SELU(
    alpha: Float = 1.6732632423543772848170429916717f,
    scale: Float = 1.0507009873554804934193349852946f
) : GradAggregateActivation({
    scale * elu(alpha, it)
})

class GELU : GradAggregateActivation({
    0.5f * it * (1.0f + Tensor.tanh(sqrt(2.0f / PI).toFloat() * (it + (0.044715f * (it pow 3)))))
})

class Softplus(
    beta: Float = 1.0f,
    threshold: Float = 20.0f
) : Activation({
    val mask = it.noGrad() gt threshold
    val index = mask.argMax(axis = -1, keepDimensions = true)
    val clampedRes = 1.0f / beta * Tensor.log(1.0f + Tensor.exp(beta * it.clamp(max = threshold)))
    clampedRes.scatter(axis = -1, index, it)
}) //TODO test me (is clamp should be called on the scaled result instead?, isn't 20 too high?, or even scattering by argmax is ok?)

class PReLU(
    size: Int = 1, //TODO size check in postInvoke (why is this an Int parameter???)
    init: Initializer = constant(0.25f)
) : Activation({ it }) {

    var weight by parameters

    init {
        weight = init.init(listOf(size)).toPlatform(platform.platformKey).asVariable(true)
    }

    override fun forward(input: Tensor): Tensor {
        return input.clamp(min = 0.0f) + weight * input.clamp(max = 0.0f)
    }
}

class Softsign : Activation({ it / (1f + Tensor.abs(it)) })

