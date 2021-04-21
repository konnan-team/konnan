package eu.redbean.konnan.optimizers.schedulers

import eu.redbean.kten.api.tensor.pow
import kotlin.math.pow

fun interface LearningRateScheduler {

    fun getLearningRate(iteration: Int, initialLR: Float, currentLR: Float, currentLoss: Float): Float

}

interface HasSerializableState {
    fun getState(): Map<String, Number>
    fun setState(optimizerState: Map<String, Number>)
}

class FixedLR(): LearningRateScheduler {
    override fun getLearningRate(iteration: Int, initialLR: Float, currentLR: Float, currentLoss: Float): Float {
        return currentLR
    }
}

class LinearDecay(
    private val decay: Float,
    private val minimum: Float = 0f
): LearningRateScheduler {

    override fun getLearningRate(iteration: Int, initialLR: Float, currentLR: Float, currentLoss: Float): Float {
        val res = initialLR * (1.0f / (1.0f + decay * iteration))
        if (res < minimum)
            return minimum
        return res
    }

}

class ExponentialDecay(
    private val decay: Float,
    private val iterationScale: Int = 100,
    private val minimum: Float = 0f
): LearningRateScheduler {

    override fun getLearningRate(iteration: Int, initialLR: Float, currentLR: Float, currentLoss: Float): Float {
        val res = initialLR * decay.pow(1f + iteration.toFloat() / iterationScale)
        if (res < minimum)
            return minimum
        return res
    }

}

class FixedSchedule(
    vararg schedule: Pair<Int, Float>
): LearningRateScheduler {

    private val scheduleList: MutableList<Pair<IntRange, Float>>

    init {
        val sortedSchedule = schedule.sortedBy { it.first }
        scheduleList = mutableListOf()
        for (i in 0 until sortedSchedule.size) {
            if (i == 0)
                scheduleList.add(0..sortedSchedule[i].first to -1f)
            else if (i + 1 < sortedSchedule.size)
                scheduleList.add(sortedSchedule[i].first..sortedSchedule[i + 1].first to sortedSchedule[i].second)
            else
                scheduleList.add(sortedSchedule[i].first..Int.MAX_VALUE to sortedSchedule[i].second)
        }
    }

    override fun getLearningRate(iteration: Int, initialLR: Float, currentLR: Float, currentLoss: Float): Float {
        val res = scheduleList.find { it.first.contains(iteration) }?.second ?: -1f
        if (res < 0f)
            return initialLR
        return res
    }

}

class DecayOnPlateau(
    private val decay: Float = 0.1f,
    private val patience: Int = 200,
    threshold: Float = 1e-4f,
    private val minimum: Float = 0f
): LearningRateScheduler {

    private var best: Float = Float.POSITIVE_INFINITY
    private var badIterations = 0
    private val relEpsilon = 1f - threshold

    override fun getLearningRate(iteration: Int, initialLR: Float, currentLR: Float, currentLoss: Float): Float {
        var res = currentLR

        if (currentLoss < best * relEpsilon) {
            badIterations = 0
            best = currentLoss
        } else {
            badIterations++
        }

        if (badIterations > patience) {
            badIterations = 0
            best = currentLoss
            res *= decay
        }

        if (res < minimum)
            return minimum

        return res
    }

}

