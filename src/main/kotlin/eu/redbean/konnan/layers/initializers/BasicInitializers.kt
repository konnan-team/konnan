package eu.redbean.konnan.layers.initializers

import eu.redbean.kten.api.autograd.utils.toStoreSize
import eu.redbean.kten.api.tensor.*
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.Tensor.Companion.sum
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.truncate
import kotlin.random.Random

fun uniform(min: Double = 0.0, max: Double = 1.0) = Initializer { shape ->
    Tensor(shape) { Random.nextDouble(min, max).toFloat() }
}

fun normal(mean: Float = 0f, std: Float = 1f) = Initializer { Tensor.randomTensor(it) * std + mean }

fun constant(value: Float = 0f) = Initializer { Tensor(it) { value } }

fun dirac() = Initializer { shape ->
    if (shape.size !in listOf(3, 4, 5)) {
        throw IllegalArgumentException("Only 3, 4 or 5 dimension tensor can be initialized with Dirac delta initializer")
    }

    val minSize = min(shape[0], shape[1])

    val res = Tensor(shape) { 0f }

    for (i in 0 until minSize) {
        if (shape.size == 3) {
            res[i, i, shape[2] / 2] = 1f
        } else if (shape.size == 4) {
            res[i, i, shape[2] / 2, shape[3] / 2] = 1f
        } else if (shape.size == 5) {
            res[i, i, shape[2] / 2, shape[3] / 2, shape[4] / 2] = 1f
        }
    }

    res
}

private fun fanInFanOut(shape: List<Int>): Pair<Int, Int> {
    if (shape.size < 2)
        throw IllegalArgumentException("Fan-in, fan-out cannot be calculated for less than two dimensions")

    val fanIn = shape[1]
    val fanOut = shape[0]
    val receptiveFieldSize = if (shape.size > 2) shape.drop(2).toStoreSize() else 1
    return fanIn * receptiveFieldSize to fanOut * receptiveFieldSize
}

fun xavierUniform(gain: Double = 1.0) = Initializer { shape ->
    val (fanIn, fanOut) = fanInFanOut(shape)
    val std = gain * sqrt(2.0 / (fanIn + fanOut))
    val bound = sqrt(3.0) * std
    uniform(-bound, bound).init(shape)
}

fun xavierNormal(gain: Double = 1.0) = Initializer { shape ->
    val (fanIn, fanOut) = fanInFanOut(shape)
    val std = gain * sqrt(2.0 / (fanIn + fanOut))
    normal(0f, std.toFloat()).init(shape)
}

fun heUniform(negativeSlope: Double = 0.0, fanInMode: Boolean = true) = Initializer { shape ->
    val (fanIn, fanOut) = fanInFanOut(shape)
    val fan = (if (fanInMode) fanIn else fanOut).toDouble()
    val gain = sqrt(2.0 / (1.0 + negativeSlope * negativeSlope))
    val std = gain / sqrt(fan)
    val bound = sqrt(3.0) * std
    uniform(-bound, bound).init(shape)
}

fun heNormal(negativeSlope: Double = 0.0, fanInMode: Boolean = true) = Initializer { shape ->
    val (fanIn, fanOut) = fanInFanOut(shape)
    val fan = (if (fanInMode) fanIn else fanOut).toDouble()
    val gain = sqrt(2.0 / (1.0 + negativeSlope * negativeSlope))
    val std = gain / sqrt(fan)
    normal(0f, std.toFloat()).init(shape)
}

private fun householderQRDecompose(matrix: Tensor): Pair<Tensor, Tensor> {

    fun householder(x: Tensor): Pair<Tensor, Float> {
        val alpha = x[0]
        val s = sum(x[1..-1] pow 2)
        var v = x.copy()

        val tau: Float
        if (s.item() == 0f) {
            tau = 0f
        } else {
            val t = Tensor.sqrt((alpha pow 2) + s)
            v[0] = if (alpha.item() <= 0f) alpha - t else -s / (alpha + t)
            tau = 2 * (v[0] pow 2).item() / (s + (v[0] pow 2)).item()
            v /= v[0]
        }

        return v to tau
    }

    if (matrix.dimensions != 2) {
        throw IllegalArgumentException("Cannot Q R decompose tensor with shape: ${matrix.shape}, only 2D tensors allowed")
    }

    val (m, n) = matrix.shape

    var mR = matrix.copy()
    var mQ = Tensor.eye(m)

    for (k in 0 until n-1) {
        val (v, tau) = householder(mR[k..-1, k..k].squeeze(-1))
        val mH = Tensor.eye(m)
        mH[k..-1, k..-1] -= truncate(tau) * (v.reshape(-1, 1) matmul v.unsqueeze(0))
        mR = mH matmul mR
        mQ = mH matmul mQ
    }

    return mQ[0 until n].transpose(0, 1) to mR[0 until n]
}

fun orthogonal(gain: Double = 1.0) = Initializer { shape ->
    if (shape.size < 2)
        throw IllegalArgumentException("Orthogonal initialization can only be applied for shapes with at least two dimensions")

    val rows = shape[0]
    val cols = shape.drop(1).toStoreSize()

    val rowsChanged = if (rows < cols) cols else rows

    val flattened = normal().init(listOf(rowsChanged, cols))

    val (q, r) = householderQRDecompose(flattened)
    val diagSize = min(r.shape[0], r.shape[1])
    val d = Tensor(diagSize) { r[it, it].item() }
    val ph = d.sign()
    val qHat = q * ph.expand(q.shape)

    qHat[0 until rows, 0 until cols].reshape(shape) * gain.toFloat()
}