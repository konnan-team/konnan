package eu.redbean.konnan.layers

import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.autograd.utils.normalizeAxis
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.serialization.TensorSerializer

class BatchNormalization(
    val axis: Int,
    val momentum: Float = 0.99f,
    val epsilon: Float = 1e-3f,
    val center: Boolean = true,
    val scale: Boolean = true,
    name: String? = null
): Layer(name) {

    private var normAxis = axis

    var gamma: Tensor by parameters
    var beta: Tensor by parameters

    var movingMean = Tensor.zeros(1)
    var movingVar = Tensor.zeros(1)

    override fun postInvoke() {
        normAxis = shape.normalizeAxis(axis)
        val paramShape = listOf(1) + shape.mapIndexed { index, size -> if (index == normAxis) size else 1 }

        if (scale)
            gamma = Tensor.ones(*paramShape.toIntArray()).toPlatform(platform.platformKey).asVariable(requiresGrad = true)

        if (center)
            beta = Tensor.zeros(*paramShape.toIntArray()).toPlatform(platform.platformKey).asVariable(requiresGrad = true)
    }

    override fun platformChanged() {
        super.platformChanged()
        movingMean = movingMean.toPlatform(platform.platformKey)
        movingVar = movingVar.toPlatform(platform.platformKey)
    }

    private fun updateMovingMeanAndVar(mean: Tensor, variance: Tensor) {
        val origMovingMean = movingMean
        val origMovingVar = movingVar
        val currentMeanMoment = mean.noGrad() * (1f - momentum)
        val currentVarMoment = variance.noGrad() * (1f - momentum)
        if (movingMean.shape != mean) {
            movingMean = currentMeanMoment
            movingVar = currentVarMoment
        } else {
            movingMean = movingMean * momentum + currentMeanMoment
            movingVar = movingVar * momentum + currentVarMoment
        }

        // ensure to survive garbage collection
        origMovingMean.release()
        origMovingVar.release()
        movingMean.incrementRef()
        movingVar.incrementRef()
    }

    override fun forward(input: Tensor): Tensor {
        var res: Tensor
        if (training) {
            res = (input as AGTensor).gradientAggregate {
                var mean = it.mean(axis = 0, keepDimensions = true) //TODO implement multiple axes version, and replace this
                var variance = it.variance(axis = 0, keepDimensions = true)
                for (a in 0 until shape.size)
                    if (a != normAxis) {
                        mean = mean.mean(axis = a + 1, keepDimensions = true)
                        variance = variance.variance(axis = a + 1, keepDimensions = true)
                    }
                updateMovingMeanAndVar(mean, variance)
                var result = (it - mean) / Tensor.sqrt(variance + epsilon)
                if (scale)
                    result *= gamma

                if (center)
                    result += beta

                result
            }
        } else {
            res = (input - movingMean) / Tensor.sqrt(movingVar + epsilon)

            if (scale)
                res *= gamma

            if (center)
                res += beta
        }


        return res
    }

    override fun <T> serializeParametersWith(serializer: TensorSerializer<T>, prefix: String): List<Pair<String, T>> {
        return super.serializeParametersWith(serializer, prefix)
            .plus((prefix + "moving_mean" to movingMean.serializeWith(serializer)))
            .plus((prefix + "moving_var" to movingVar.serializeWith(serializer)))
    }

    override fun <T> deserializeParametersWith(serializer: TensorSerializer<T>, serializedParametes: List<Pair<String, T>>) {
        //TODO create a more generic solution for these (@UnoptimizedParameter annotation or something similar)
        movingMean = Tensor.deserializeWith(serializer, serializedParametes.find { it.first == "moving_mean" }!!.second).toPlatform(platform.platformKey)
        movingVar = Tensor.deserializeWith(serializer, serializedParametes.find { it.first == "moving_var" }!!.second).toPlatform(platform.platformKey)

        super.deserializeParametersWith(serializer, serializedParametes.filter { it.first !in listOf("moving_mean", "moving_var") })
    }
}