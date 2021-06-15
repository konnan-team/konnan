package eu.redbean.konnan.test

import eu.redbean.konnan.layers.Conv2D
import eu.redbean.konnan.layers.Dense
import eu.redbean.konnan.layers.Flatten
import eu.redbean.konnan.layers.Input
import eu.redbean.konnan.layers.activations.ReLU
import eu.redbean.konnan.layers.activations.Softmax
import eu.redbean.konnan.layers.initializers.constant
import eu.redbean.konnan.layers.initializers.heNormal
import eu.redbean.konnan.models.Model
import eu.redbean.konnan.optimizers.AbstractOptimizer
import eu.redbean.konnan.optimizers.Adam
import eu.redbean.konnan.optimizers.schedulers.LinearDecay
import eu.redbean.konnan.optimizers.utils.clipGradientNormalizer
import eu.redbean.kten.api.tensor.*
import eu.redbean.kten.api.tensor.Constants.all
import eu.redbean.kten.api.tensor.Tensor.Companion.abs
import eu.redbean.kten.api.tensor.Tensor.Companion.sqrt
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution
import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.QuickChart
import java.nio.file.Path
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.stream.IntStream
import kotlin.concurrent.thread
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sign
import kotlin.random.Random


class Rect(
    var x: Int, var y: Int,
    var width: Int, var height: Int,
    var value: Float
) {

    fun draw(canvas: Tensor) {
        val left = min(max(x, 0), canvas.shape[2] - 1)
        val top = min(max(y, 0), canvas.shape[1] - 1)
        val right = min(max(x + width, 1), canvas.shape[2])
        val bottom = min(max(y + height, 1), canvas.shape[1])
        canvas[0..0, top until bottom, left until right] = value
    }

    fun intersects(other: Rect): Boolean {
        return x + width >= other.x && x <= other.x + other.width && y + height >= other.y && y <= other.y + other.height
    }

}

class Brick(
    val column: Int,
    val row: Int,
    val value: Float,
    val gridWidth: Int = 12
) {

    val gridOffsetTop = 23
    val gap = 0
    val brickHeight = 2
    val brickWidth = (84 - gap * gridWidth) / gridWidth

    val rect = Rect(
        column * brickWidth + (column + 1) * gap,
        gridOffsetTop + brickHeight * row + row * gap,
        brickWidth, brickHeight,
        value
    )

    fun hit(ball: Ball): Boolean {
        if (rect.value == 0f)
            return false

        val hit = rect.intersects(ball.rect)
        if (hit)
            rect.value = 0f
//            rect.value -= 0.2f // replace the previous line to this, if you'd like to make the game more difficult

        if (rect.value < 0.2f) {
            rect.value = 0f
        }

        return hit
    }

    fun draw(canvas: Tensor) {
        if (rect.value > 0f)
            rect.draw(canvas)
    }

}

class Ball(
    var position: Tensor
) {
    val width = 1
    val height = 1

    val direction = Tensor.tensorOf(1f, Random.nextFloat()*1.2f - 0.6f)
    var speed = 1f

    var active = false

    val rect = Rect(
        position[1].item().roundToInt(), position[0].item().roundToInt(), width, height, 1f
    )

    init {
        normalizeDirection()
    }

    fun normalizeDirection() {
        direction[1] = direction[1].clamp(min = -0.8f, max = 0.8f)
        if (abs(direction[1]).item() < 0.05f)
            direction[1] = 0.05f * direction[1].item().sign
        if (direction[1].item() == 0f)
            direction[1] = 0.05f
        direction[0] = direction[0] * sqrt(1f - abs(direction[1]))
    }

    fun draw(env: Environment) {
        if (active) {
            val speedInt = speed.roundToInt()
            for (i in 0 until speedInt) {
                updatePosition()
                updateDirection(env)
            }
            rect.draw(env.canvas)
        }
    }

    fun updatePosition() {
        position += direction
        rect.y = position[0].item().toInt()
        rect.x = position[1].item().toInt()
    }

    fun updateDirection(env: Environment) {
        if (rect.x < 0 || (rect.x + rect.width) > env.canvas.shape[2]) {
            direction[1] = direction[1] * -1f
            if (rect.x < 0)
                rect.x = 0
            else if (rect.x + rect.width > env.canvas.shape[2])
                rect.x = env.canvas.shape[2] - rect.width
        }
        if (rect.y < 0) {
            direction[0] = direction[0] * -1f
            rect.y = 0
        }

        if (rect.y >= env.canvas.shape[1]) {
            reset(env)
            return
        }

        val y = rect.y
        val x = rect.x

        var changedY = false

        env.grid.hitDetect(this).forEach { brickHit ->
            if ((x + rect.width == brickHit.rect.x
                        || x == brickHit.rect.x + brickHit.rect.width)
                && y + rect.height > brickHit.rect.y
                && y < brickHit.rect.y + brickHit.rect.height) {
                direction[1] = direction[1] * -1f
            }
            if ((y + rect.height == brickHit.rect.y || y == brickHit.rect.y + brickHit.rect.height)
                && x + rect.width > brickHit.rect.x
                && x < brickHit.rect.x + brickHit.rect.width && !changedY) {
                direction[0] = direction[0] * -1f
                changedY = true;
            }

            env.score += 1
            speed += 0.02f
            if (speed > 3f)
                speed = 3f

            if (brickHit.rect.value == 0f) {
                val brickValue = (brickHit.value * 6).toInt()
                env.score += brickValue - 1
                if (speed < 2f && brickValue > 3f)
                    speed = 2f
            }
        }

        if (env.paddle.rect.intersects(rect)) {
            direction[0] = -1f
            val paddleCenter = env.paddle.rect.x + env.paddle.rect.width / 2f
            val ballCenter = rect.x + rect.width / 2f
            val dist = ballCenter - paddleCenter
            val maxDist = env.paddle.rect.width / 2f + 1f
            direction[1] = dist / maxDist
            normalizeDirection()
        }
    }

    fun reset(env: Environment) {
        env.lifes -= 1
        position[0] = env.canvas.shape[1] - 40f
        position[1] = env.canvas.shape[2] / 2
        rect.x = position[1].item().roundToInt()
        rect.y = position[0].item().roundToInt()
        speed = 1f
        direction[0] = 1f
        direction[1] = Random.nextFloat()*1.2f - 0.6f
        normalizeDirection()
        active = false
    }

}

class Grid(
    val rows: Int,
    val columns: Int
) {

    val bricks = List(rows * columns) {
        Brick(
            it % columns,
            it / columns,
            (rows - it / columns).toFloat() / rows * 0.8f + 0.2f,
            columns
        )
    }

    fun draw(env: Environment) {
        bricks.forEach { it.draw(env.canvas) }
    }

    fun hitDetect(ball: Ball): List<Brick> = bricks.filter { it.hit(ball) }

}

class Paddle(
    y: Int,
    x: Int
) {

    val rect = Rect(x, y, 8, 1, 1f)

    private val directionCap = rect.width / 3f

    var internalDirection = 0f

    fun move(direction: Int, env: Environment) {
        internalDirection += (directionCap * direction) / 4f
        if (kotlin.math.abs(internalDirection) > directionCap) {
            internalDirection = internalDirection.sign * directionCap
        }

        val intDir = internalDirection.toInt()

        if (direction == 0 && intDir != 0) {
            internalDirection += -internalDirection.sign * (directionCap / 4f)
            rect.x = rect.x + internalDirection.toInt()
        }

        if (direction != 0 && (direction.sign == intDir.sign)) {
            rect.x = rect.x + intDir
        }

        val widthHalf = (rect.width / 2) - 1
        if (rect.x < 0) {
            rect.x = 0
        } else if (rect.x > env.canvas.shape[2] - widthHalf) {
            rect.x = env.canvas.shape[2] - widthHalf
        }

        rect.draw(env.canvas)
    }
}

class Environment(
    val canvas: Tensor
) {
    var paddle = Paddle(canvas.shape[1] - 8, canvas.shape[2] / 2 - 4)
    var ball = Ball(Tensor.tensorOf(canvas.shape[1] - 40, canvas.shape[2] / 2))
    var grid = Grid(6, 12)
    var score = 0
    var lifes = 5
    val lifeIndicator = Rect(canvas.shape[1] - (lifes*5), 0, lifes*5, 3, 0.5f)
    var scoreBefore = 0
    var sumEpScore = 0

    val sumScore get() = sumEpScore + score

    fun step(action: Int): Pair<Float, Boolean> {
        val lifesBefore = lifes
        canvas[all, all, all] = 0f
        val direction = when (action) {
            0 -> 0
            1 -> 0
            2 -> -1
            3 -> 1
            4 -> -3
            5 -> 3
            else -> throw IllegalArgumentException("Invalid action")
        }
        if (action == 1) {
            ball.active = true
        }
        paddle.move(direction, this)
        ball.draw(this)
        grid.draw(this)
        val reward = (score - scoreBefore).toFloat()
        scoreBefore = score

        if (ball.rect.y > canvas.shape[1] / 2 && grid.bricks.none { it.rect.value > 0 }) {
            grid.bricks.forEach {
                it.rect.value = it.value
            }
        }

        if (lifesBefore > lifes) {
            lifeIndicator.x = canvas.shape[1] - (lifes*5)
            lifeIndicator.width = lifes*5
        }
        lifeIndicator.draw(canvas)

        if (lifes == 0) {
            return 0f to true
        }
        return reward to false
    }

    fun reset() {
        paddle = Paddle(canvas.shape[1] - 8, canvas.shape[2] / 2 - 4)
        ball.reset(this)
        grid = Grid(6, 12)
        sumEpScore += score
        score = 0
        scoreBefore = 0
        lifes = 5
        lifeIndicator.x = canvas.shape[1] - (lifes*5)
        lifeIndicator.width = lifes*5
    }
}

class Policy(
    val observationShape: List<Int>,
    val actionSpace: Int,
    val optimizer: AbstractOptimizer = Adam(),
    val clipRange: Float = 0.2f,
    val entropyCoeff: Float = 0.01f,
    val vfCoeff: Float = 0.5f,
    val platformKey: String = PlatformProvider.defaultPlatformKey
) {

    val observationInput = Input(*observationShape.toIntArray())
    val oldProbInput = Input(actionSpace)
    val oldValueInput = Input(1)
    val advantageInput = Input(1)
    val policyOut = Dense(actionSpace)
    val valueOut = Dense(1)

    val fullModel: Model
    val mainModel: Model

    init {
        var layer = Conv2D(32, 8, 4, weightInitializer = heNormal(), biasInitializer = constant(0f))(observationInput)
        layer = ReLU()(layer)
        layer = Conv2D(64, 4, 2, weightInitializer = heNormal(), biasInitializer = constant(0f))(layer)
        layer = ReLU()(layer)
        layer = Conv2D(32, 3, 1, weightInitializer = heNormal(), biasInitializer = constant(0f))(layer)
        layer = ReLU()(layer)
        layer = Flatten()(layer)
        var fc1 = Dense(512, weightInitializer = heNormal(), biasInitializer = constant(0f))(layer)
        fc1 = ReLU()(fc1)
        val policyOutput = Softmax()(policyOut(fc1))
        val valueOutput = valueOut(fc1)

        mainModel = Model(listOf(observationInput), listOf(policyOutput, valueOutput))
        mainModel.onPlatform(platformKey)

        fullModel = Model(listOf(observationInput, advantageInput, oldProbInput, oldValueInput), listOf(policyOutput, valueOutput))
        fullModel.onPlatform(platformKey)
        fullModel.summary()
        fullModel.gradNormalizer = clipGradientNormalizer(0.5f)
        fullModel.prepare(optimizer, this::policyLoss, this::valueFunctionLoss)
    }

    fun policyLoss(yPred: Tensor, yTrue: Tensor): Tensor {
        val negLogProbDiff = -(yTrue * Tensor.log(oldProbInput.value)).sum(axis = -1, keepDimensions = true) - (-(yTrue * Tensor.log(yPred)).sum(axis = -1, keepDimensions = true))
        val ratio = Tensor.exp(negLogProbDiff)
        val pgLoss1 = -advantageInput.value * ratio
        val pgLoss2 = -advantageInput.value * ratio.clamp(1.0f - clipRange, 1.0f + clipRange)
        val mask = pgLoss1.noGrad() gt pgLoss2.noGrad()
        val pgLoss = Tensor.mean((pgLoss1 * mask) + (pgLoss2 * (1f - mask)))
        val entropyLoss = Tensor.mean(-((yPred * Tensor.log(yPred)).sum(axis = -1)))
        return pgLoss - (entropyCoeff * entropyLoss)
    }

    fun valueFunctionLoss(yPred: Tensor, yTrue: Tensor): Tensor {
        val yPredClipped = oldValueInput.value + (yPred - oldValueInput.value).clamp(-clipRange, clipRange)
        val vfLoss1 = (yTrue - yPred) pow 2
        val vfLoss2 = (yTrue - yPredClipped) pow 2
        val mask = vfLoss1.noGrad() gt vfLoss2.noGrad()
        return vfCoeff * Tensor.mean(vfLoss1 * mask + vfLoss2 * (1f - mask))
//        return vfCoeff * mse(yPred, yTrue)
    }

}

class Agent(
    val policy: Policy,
    val actionSpace: Int,
    val observationShape: List<Int>,
    val numberOfEnvs: Int = 8,
    val rolloutLength: Int = 10,
    val rewardScale: Float = 1f,
    val gamma: Float = 0.99f,
    val gaeLambda: Float = 0.95f,
    val normalizeAdvantages: Boolean = true,
    val epochs: Int = 3,
    val minibatchSize: Int = 64,
    var epsilon: Float = -1f
) {

    private val EYE = Tensor.eye(actionSpace)

    var batchObs = Tensor.zeros(1)

    val environments = List(numberOfEnvs) {
        val singleEnvShape = observationShape.toIntArray()
        singleEnvShape[0] /= 4
        Environment(Tensor.zeros(*singleEnvShape))
    }

    lateinit var batchProbs: Tensor
    lateinit var batchActions: Tensor
    lateinit var batchValues: Tensor
    lateinit var batchRewards: Tensor
    lateinit var batchNotDones: Tensor
    lateinit var lastVals: Tensor
    lateinit var lastObs: Tensor

    val topScore = AtomicInteger(0)
    val sumScore = AtomicInteger(0)
    val meanReturnsLog = mutableListOf<Double>()
    val entropyLog = mutableListOf<Double>()


    private fun choose(probabilities: Tensor): IntArray {
        val batch = probabilities.shape[0]
        if (epsilon >= 0f) {
            // Epsilon greedy option. (Normally the entropy bonus is enough for exploration,
            // but if you'd like to check the configuration with a fixed random action probability
            // you can provide an epsilon value in 0f..1f range)
            val actions = probabilities.argMax(axis = -1)
            return IntArray(batch) { b ->
                if (Random.nextFloat() < epsilon)
                    Random.nextInt(actionSpace)
                else actions[b].item().toInt()
            }
        }

        return IntArray(batch) { b ->
            val choices = IntArray(probabilities.shape[1]) { it }
            val probDist = DoubleArray(probabilities.shape[1]) { probabilities[b, it].item().toDouble() }
            val distribution = EnumeratedIntegerDistribution(choices, probDist)
            distribution.sample()
        }
    }

    private fun oneHot(actions: IntArray): Tensor {
        return Tensor.concat(List(actions.size) { EYE[actions[it]].unsqueeze(0) })
    }

    fun step() {
        if (batchObs.dimensions == 1) { //bootstrap
            lastObs = stepInEnvs(IntArray(numberOfEnvs)).first
        }

        val probsList = mutableListOf<Tensor>()
        val actionsList = mutableListOf<Tensor>()
        val obsList = mutableListOf<Tensor>()
        val notDonesList = mutableListOf<Tensor>()
        val rewsList = mutableListOf<Tensor>()
        val valsList = mutableListOf<Tensor>()

        for (i in 0 until rolloutLength) {
            val (probs, values) = policy.mainModel.predictOnBatch(listOf(lastObs))
            val actions = choose(probs)
            val (envsObs, envsNotDones, envRewards) = stepInEnvs(actions)
            val envActions = oneHot(actions)

            probsList += probs
            actionsList += envActions
            obsList += lastObs
            notDonesList += envsNotDones
            rewsList += envRewards
            valsList += values

            lastObs = envsObs
        }

        val avgScore = sumScore.get().toFloat() / numberOfEnvs

        println("Full rollout is done top score: ${topScore.get()} avg, score: $avgScore")
        sumScore.set(0)

        batchProbs = Tensor.concat(probsList)
        batchObs = Tensor.concat(obsList)
        batchActions = Tensor.concat(actionsList)
        batchRewards = Tensor.concat(rewsList.map { it.unsqueeze(axis = 0) })
        batchValues = Tensor.concat(valsList.map { it.unsqueeze(axis = 0) })
        batchNotDones = Tensor.concat(notDonesList.map { it.unsqueeze(axis = 0) })

        val (_, values) = policy.mainModel.predictOnBatch(listOf(lastObs))
        lastVals = values
    }

    private fun stepInEnvs(actions: IntArray): Triple<Tensor, Tensor, Tensor> {
        val obs = mutableListOf<Tensor>()
        val dones = mutableListOf<Boolean>()
        val rewards = mutableListOf<Float>()
        val envMap = ConcurrentHashMap(mutableMapOf<Int, Triple<Tensor, Boolean, Float>>())
        IntStream.range(0, environments.size).parallel().forEach { index ->
            val env = environments[index]
            var rewSum = 0f
            var doneSum = false
            var realDone = false
            val repetedStepObs = mutableListOf<Tensor>()
            for (i in 0 until 4) {
                if (!realDone) {
                    val lifesBefore = env.lifes
                    val (reward, done) = env.step(actions[index])
                    rewSum += reward.sign
                    if (done || env.lifes < lifesBefore)
                        doneSum = true

                    realDone = done
                }
                repetedStepObs += env.canvas
            }
            topScore.getAndUpdate { if (it < env.score) env.score else it }
            if (realDone) {
                sumScore.addAndGet(env.score)
                env.reset()
                env.step(0)
                repetedStepObs.clear()
                val zeroObs = Tensor.zerosLike(env.canvas)
                repetedStepObs += listOf(zeroObs, zeroObs, zeroObs, env.canvas)
            }

            envMap[index] = Triple(Tensor.concat(repetedStepObs).unsqueeze(axis = 0), doneSum, rewSum)
        }
        for (e in environments.indices) {
            val (localObs, localDones, localRewards) = envMap[e]!!
            obs.add(localObs)
            dones.add(localDones)
            rewards.add(localRewards)
        }
        val envsObs = Tensor.concat(obs)
        val envsNotDones = Tensor(dones.size) { if (dones[it]) 0f else 1f }.unsqueeze(axis = -1)
        val envRewards = Tensor(rewards.size) { rewards[it] }.unsqueeze(axis = -1)
        return Triple(envsObs, envsNotDones, envRewards)
    }

    fun generalizedAdvantageEstimator(): Pair<Tensor, Tensor> {
        var advantages = Tensor.zerosLike(batchRewards)
        var nextAdvantages = Tensor.zeros(1)
        var nextVals: Tensor
        for (t in rolloutLength - 1 downTo 0) {
            if (t == rolloutLength - 1) {
                nextVals = lastVals
            } else {
                nextVals = batchValues[t+1]
            }
            val delta = rewardScale * batchRewards[t] + gamma * batchNotDones[t] * nextVals - batchValues[t]
            nextAdvantages = delta + gamma * gaeLambda * batchNotDones[t] * nextAdvantages
            advantages[t] = nextAdvantages
        }

        val returns = (advantages + batchValues).reshape(-1, 1)
        if (normalizeAdvantages) {
            advantages = advantages.reshape(-1)
            advantages = ((advantages - advantages.mean()) / (advantages.std(0) + 1e-10f)).unsqueeze(axis = -1)
        } else {
            advantages = advantages.reshape(-1, 1)
        }

        return advantages to returns
    }

    fun bootstrapEnvs(bootstrapSteps: Int = 5000) {
        val sumScore = AtomicInteger(0)
        for (i in 0 until bootstrapSteps) {
            environments.stream().parallel().forEach {
                val (_, done) = it.step(Random.nextInt(actionSpace))
                sumScore.addAndGet(it.score)
                if (done)
                    it.reset()
                it.sumEpScore = 0
            }
        }
        println("Bootstrap done, avg score for random steps: ${sumScore.get().toDouble() / (bootstrapSteps * environments.size)}")
    }

    fun train() {
        fun rearrange(tensor: Tensor, indices: List<Int>) = Tensor.concat(indices.map { tensor[it].unsqueeze(0) }, axis = 0)

        println("Training start, current learning rate: ${policy.optimizer.currentLR}")
        val (advantages, returns) = generalizedAdvantageEstimator()
        val meanReturn = environments.map { it.sumScore }.reduce(Int::plus).toDouble() / numberOfEnvs
        environments.forEach { it.sumEpScore = 0 }
        meanReturnsLog += meanReturn
        println("Mean of returns: ${meanReturn}")

        val currentEntropy = Tensor.mean(-((batchProbs * Tensor.log(batchProbs)).sum(axis = -1))).item().toDouble()
        println("Current entropy: $currentEntropy")
        entropyLog += currentEntropy

        batchValues = batchValues.reshape(-1, 1)
        val steps = batchObs.shape[0] / minibatchSize
        val lastStepSize = batchObs.shape[0] % minibatchSize

        for (epoch in 0 until epochs) {
            val shuffledIndices = (0 until batchValues.shape[0]).shuffled()
            val obs = rearrange(batchObs, shuffledIndices)
            val advs = rearrange(advantages, shuffledIndices)
            val probs = rearrange(batchProbs, shuffledIndices)
            val vals = rearrange(batchValues, shuffledIndices)
            val acts = rearrange(batchActions, shuffledIndices)
            val rets = rearrange(returns, shuffledIndices)

            val epochMetrics = mutableMapOf<String, Float>()
            for (step in 0 until steps) {
                val minibatchRange =  (step*minibatchSize) until (step*minibatchSize + minibatchSize)
                val metrics = policy.fullModel.trainOnBatch(
                    listOf(obs[minibatchRange], advs[minibatchRange], probs[minibatchRange], vals[minibatchRange]),
                    listOf(acts[minibatchRange], rets[minibatchRange])
                )
                if (epochMetrics.isEmpty()) {
                    epochMetrics.putAll(metrics)
                } else {
                    metrics.forEach { key, value -> epochMetrics[key] = epochMetrics[key]!! + value }
                }
            }
            if (lastStepSize > 0) {
                val minibatchRange =  (steps*minibatchSize) until (steps*minibatchSize + lastStepSize)
                val metrics = policy.fullModel.trainOnBatch(
                    listOf(obs[minibatchRange], advs[minibatchRange], probs[minibatchRange], vals[minibatchRange]),
                    listOf(acts[minibatchRange], rets[minibatchRange])
                )
                if (epochMetrics.isEmpty()) {
                    epochMetrics.putAll(metrics)
                } else {
                    metrics.forEach { key, value -> epochMetrics[key] = epochMetrics[key]!! + value }
                }
            }
            val metrics = epochMetrics.map { (key, value) -> key to (value / if (lastStepSize > 0) steps + 1 else steps) }.toMap()
            println("Training metrics: $metrics")
        }
        topScore.set(0)
    }
}

class FramesCalculator(
    val rolloutLength: Int = 128,
    val numberOfEnvs: Int = 16,
    val minibatchSize: Int = 256,
    val trainEpochs: Int = 4
) {
    val framesPerEpisode = numberOfEnvs * rolloutLength
    fun calcTrainingStepsForMaxFrames(maxFrames: Int) = maxFrames * trainEpochs / minibatchSize
    fun calcMaxEpisodes(maxFrames: Int) = maxFrames / framesPerEpisode
}

fun main() {
    val loadWeights = true
    val train = false
    val play = true
    val platformKey = "OpenCL - 2 - AMD Radeon Pro 455 Compute Engine" // change to your preferred platform

    val framesCalculator = FramesCalculator(
        rolloutLength = 128,
        numberOfEnvs = 16,
        trainEpochs = 4
    )
    val learningRate = 1.0e-4f
    val maxFrames = 10_000_000

    val policy = Policy(listOf(4, 84, 84), 4,
        platformKey = platformKey,
        vfCoeff = 0.5f,
        optimizer = Adam(learningRate, scheduler = LinearDecay(learningRate / framesCalculator.calcTrainingStepsForMaxFrames(maxFrames))),
        entropyCoeff = 0.01f,
        clipRange = 0.1f)

    if (loadWeights)
        policy.fullModel.loadParams(Path.of("./brakout_ppo.weights"))

    if (train) {
        val agent = Agent(
            policy, 4,
            listOf(4, 84, 84),
            rolloutLength = framesCalculator.rolloutLength,
            numberOfEnvs = framesCalculator.numberOfEnvs,
            epochs = framesCalculator.trainEpochs,
            gaeLambda = 0.7f,
            normalizeAdvantages = true,
            minibatchSize = framesCalculator.minibatchSize
        )

        //agent.bootstrapEnvs() // may improve the first phase of training

        for (e in 0 until framesCalculator.calcMaxEpisodes(maxFrames)) {
            println("episode $e start")
            agent.step()
            agent.train()
            println("episode $e end")

            if (e % 50 == 0 && e > 0) {
                println("saving model parameters")
                policy.fullModel.saveParams(Path.of("./brakout_ppo.weights"), withOptimizerState = true)
            }
            if (e % 10 == 0 && e > 0) {
                println("plotting reward chart, and entropy chart")
                val rewardChart = QuickChart.getChart("PPO", "episode", "mean returns", null,
                    agent.meanReturnsLog.indices.toList().map { it.toDouble() }.toDoubleArray(), agent.meanReturnsLog.toDoubleArray()
                )
                BitmapEncoder.saveBitmapWithDPI(rewardChart, "./breakout_ppo_rewards", BitmapEncoder.BitmapFormat.PNG, 300)

                val entropyChart = QuickChart.getChart("PPO", "episode", "entropy", null,
                    agent.entropyLog.indices.toList().map { it.toDouble() }.toDoubleArray(), agent.entropyLog.toDoubleArray()
                )
                BitmapEncoder.saveBitmapWithDPI(entropyChart, "./breakout_ppo_entropy", BitmapEncoder.BitmapFormat.PNG, 300)
            }
        }

        policy.fullModel.saveParams(Path.of("./brakout_ppo.weights"), withOptimizerState = true)
    }

    if (play) {
        val playAgent = Agent(
            policy, 4, listOf(4, 84, 84), 1, 1, epsilon = 0.0f
        )

        var tensor = Tensor.zeros(1, 84, 84)

        val imgShow = tensor.toImage()
        imgShow.processor = imgShow.processor.resize(320)
        imgShow.show()

        while (true) {
            playAgent.step()

            tensor = playAgent.environments[0].canvas.copy()

            thread(true) {
                val img = tensor.toImage()
                synchronized(imgShow) {
                    imgShow.processor = img.processor.resize(320)
                    imgShow.updateAndDraw()
                }
            }

            Thread.sleep(50)
        }
    }
}