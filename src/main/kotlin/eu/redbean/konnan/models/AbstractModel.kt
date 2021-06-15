package eu.redbean.konnan.models

import eu.redbean.konnan.dataprocessing.Generator
import eu.redbean.konnan.layers.AbstractLayerBase
import eu.redbean.konnan.lossfunctions.LossFunction
import eu.redbean.konnan.metrics.Metric
import eu.redbean.konnan.optimizers.AbstractOptimizer
import eu.redbean.konnan.optimizers.utils.GradientNormalizer
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import java.nio.file.Path
import java.time.Duration
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.concurrent.Executors
import java.util.concurrent.ThreadFactory
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.floor

abstract class AbstractModel(
    val outputs: List<AbstractLayerBase>
) {

    protected val lossFunctions = mutableListOf<LossFunction?>()
    protected val namedLossFunctions = mutableMapOf<String, () -> Tensor>()
    protected val metrics = mutableMapOf<Int, Pair<String, Metric>>()

    protected lateinit var loss: MetricCollectingSummaryLoss
    protected lateinit var optimizer: AbstractOptimizer

    private var platform = PlatformProvider.defaultPlatformKey

    protected var prepared = false
        private set

    abstract val layers: List<AbstractLayerBase>


    fun prepare(optimizer: AbstractOptimizer, vararg loss: LossFunction?) {
        if (loss.size != outputs.size)
            throw IllegalArgumentException("Loss must be specified for all output layers in prepare call, " +
                    "output size: ${outputs.size}, loss functions size: ${loss.size}")

        this.optimizer = optimizer
        this.lossFunctions += loss
        this.loss = MetricCollectingSummaryLoss(lossFunctions, namedLossFunctions)
        prepared = true
    }

    fun onPlatform(platform: String) {
        val platformOps = PlatformProvider.tensorOperations(platform)
        outputs.forEach { it.platform = platformOps }
        this.platform = platform
    }

    fun addLoss(name: String? = null, loss: () -> Tensor) {
        if (prepared)
            throw IllegalStateException("Cannot add loss to a prepared model")

        val lossName = name ?: "custom_loss_${namedLossFunctions.size + 1}"
        namedLossFunctions[lossName] = loss
    }

    fun addMetric(metric: Metric, name: String? = null) {
        if (outputs.size != 1)
            throw IllegalArgumentException("Cannot add metric to model with non single output, without specifying the output layer.")

        addMetric(0, metric, name)
    }

    fun addMetric(outputLayer: AbstractLayerBase, metric: Metric, name: String? = null) {
        if (outputs.contains(outputLayer).not())
            throw IllegalArgumentException("Cannot add metric to non-output layers")

        addMetric(outputs.indexOf(outputLayer), metric, name)
    }

    fun addMetric(outputLayerIndex: Int, metric: Metric, name: String? = null) {
        if (outputLayerIndex < 0 || outputLayerIndex >= outputs.size)
            throw IllegalArgumentException("No output layer found for index: $outputLayerIndex")

        val metricName = name ?: "${metric.name}_${metrics.size + 1}"
        metrics[outputLayerIndex] = metricName to metric
    }

    protected fun calculateMetrics(yTrues: List<Tensor?>): Map<String, Float> {
        return metrics
            .filterKeys { yTrues[it] != null }
            .map { (index, namedMetric) -> namedMetric.first to namedMetric.second.calculate(outputs[index].value.noGrad(), yTrues[index]!!) }
            .toMap()
    }

    fun summary() {
        var parametersSum = 0
        outputs.flatMap { it.summary() }.distinct().forEach { (layerInfo, parameters) ->
            println("Layer: $layerInfo - trainable parameters: $parameters")
            parametersSum += parameters
        }

        println("Total trainable parameters: $parametersSum")
    }

    protected abstract fun internalTrainOnBatch(x: List<Tensor>, y: List<Tensor?>): Map<String, Float>

    private fun transferXYToModelPaltform(x: List<Tensor>, y: List<Tensor?>): Pair<List<Tensor>, List<Tensor?>> {
        val xTransformed = x.map {
            if (it.platform != this.platform)
                it.toPlatform(platform)
            else it
        }
        val yTransformed = y.map {
            if (it != null && it.platform != this.platform)
                it.toPlatform(this.platform)
            else it
        }

        return xTransformed to yTransformed
    }

    fun trainOnBatch(x: List<Tensor>, y: List<Tensor?>): Map<String, Float> {
        return PlatformProvider.tensorOperations(platform).garbageCollector().use {
            val (xTransformed, yTransformed) = transferXYToModelPaltform(x, y)
            internalTrainOnBatch(xTransformed, yTransformed)
        }
    }

    fun trainOnBatch(x: Tensor, y: Tensor?): Map<String, Float> {
        return trainOnBatch(listOf(x), listOf(y))
    }

    protected abstract fun internalPredictOnBatch(x: List<Tensor>): List<Tensor>

    fun predictOnBatch(x: List<Tensor>): List<Tensor> {
        return predictOnBatchToTargetPlatform(x, PlatformProvider.defaultPlatformKey)
    }

    fun predictOnBatchToTargetPlatform(x: List<Tensor>, targetPlatform: String): List<Tensor> {
        val inputPlatforms = x.map { it.platform }.distinct()

        if (inputPlatforms.size == 1 && inputPlatforms.first() == this.platform) {
            if (targetPlatform == this.platform) {
                return internalPredictOnBatch(x)
            }

            return PlatformProvider.tensorOperations(this.platform).garbageCollector().use {
                internalPredictOnBatch(x).map {
                    val res = it.toPlatform(targetPlatform)
                    it.release()
                    layers.forEach { it.output?.release() }
                    res
                }
            }
        }

        return PlatformProvider.tensorOperations(this.platform).garbageCollector().use {
            internalPredictOnBatch(x.map { it.toPlatform(this.platform) }).map {
                val res = it.toPlatform(targetPlatform)
                it.release()
                layers.forEach { it.output?.release() }
                res
            }
        }
    }

    fun predictOnBatch(x: Tensor): Tensor {
        if (outputs.size != 1)
            throw IllegalArgumentException("Model does not output single tensor, " +
                    "for single input multiple output models please use predictOnBatch(listOf(x))")

        return predictOnBatch(listOf(x))[0]
    }

    protected abstract fun tryToLoadFromCheckpoint(checkpointsPath: Path?): Pair<Int, Int>

    protected abstract fun createCheckpoint(checkpointsPath: Path, epoch: Int, step: Int): Path

    fun fitGenerator(
        epochs: Int = 1,
        steps: Int = 100,
        verbose: Boolean = true,
        checkpointsPath: Path? = null,
        checkpointFrequency: Int = 0,
        prefetchOnThreads: Int = 2,
        prefetchMaxSize: Int = 100,
        transferToPlatformInPrefetch: Boolean = true,
        generator: (step: Int) -> Pair<List<Tensor>, List<Tensor>>
    ) {
        val (epochStart, stepStart) = tryToLoadFromCheckpoint(checkpointsPath)

        if (verbose && (epochStart > 0 || stepStart > 0))
            println("Continuing from checkpoint at epoch: ${epochStart} step: ${stepStart}")

        val chkFreq = if (checkpointFrequency > 0) checkpointFrequency else steps - 1

        val buffer = ConcurrentLinkedDeque<Pair<List<Tensor>, List<Tensor?>>>()
        startPrefetching(prefetchOnThreads, prefetchMaxSize, epochs, epochStart, steps, stepStart, buffer, generator, transferToPlatformInPrefetch)
        var bufferSizeSum = 0

        val time = System.nanoTime()

        for (epoch in epochStart until epochs) {
            val stepsMetrics = mutableMapOf<String, MutableList<Float>>()
            for (step in stepStart until steps) {
                while (buffer.isEmpty())
                    Thread.sleep(100)

                val (x, y) = buffer.pop()
                val singleStepMetrics = trainOnBatch(x, y)

                if (transferToPlatformInPrefetch) {
                    x.forEach { it.release() }
                    y.forEach { it?.release() }
                }

                if (verbose) {
                    bufferSizeSum += buffer.size

                    singleStepMetrics.forEach { name, metric ->
                        stepsMetrics.putIfAbsent(name, mutableListOf())
                        stepsMetrics[name]?.add(metric)
                    }

                    val stepsBetweenReport = floor(steps / 100.0).toInt() + 1

                    if (step % stepsBetweenReport == 0) {
                        val bufferUtilization = "%.0f".format(bufferSizeSum.toDouble() / stepsBetweenReport / (prefetchMaxSize + 1) * 100.0)
                        bufferSizeSum = 0
                        val metrics = stepsMetrics.map { (name, values) ->
                            name to "%.4g".format(values.sum() / values.size)
                        }.toMap()

                        val eta = Duration.ofNanos(((System.nanoTime() - time) / (epoch * steps + step + 1)) * (epochs * steps - (epoch * steps + step)))

                        println("ETA: [${eta.toMinutes()}m ${eta.toSecondsPart()}s] Epoch: [${epoch + 1}/$epochs] Step: [${step + 1}/$steps] Loss: ${metrics["loss"]} Metrics: $metrics " +
                                "Learning rate: ${optimizer.currentLR} Buffer: $bufferUtilization%")
                        stepsMetrics.clear()
                    }

                    if (checkpointsPath != null && step > 0 && step % chkFreq == 0) {
                        val path = createCheckpoint(checkpointsPath, epoch, step)
                        if (verbose)
                            println("Weight checkpoint created at path: ${path}")
                    }
                }
            }
        }
    }

    fun fitGenerator(
        generator: Generator,
        epochs: Int = 1,
        verbose: Boolean = true,
        checkpointsPath: Path? = null,
        checkpointFrequency: Int = 0,
        prefetchOnThreads: Int = 2,
        prefetchMaxSize: Int = 100,
        transferToPlatformInPrefetch: Boolean = true,
    ) {
        fitGenerator(epochs, generator.steps, verbose, checkpointsPath, checkpointFrequency,
            prefetchOnThreads, prefetchMaxSize, transferToPlatformInPrefetch, generator::generate)
    }

    fun evaluate(
        steps: Int = 100,
        verbose: Boolean = true,
        prefetchOnThreads: Int = 2,
        prefetchMaxSize: Int = 100,
        generator: (step: Int) -> Pair<List<Tensor>, List<Tensor>>
    ): Map<String, Float> {
        if (metrics.isEmpty())
            throw IllegalStateException("No metrics specified for model to evaluate by")

        val buffer = ConcurrentLinkedDeque<Pair<List<Tensor>, List<Tensor?>>>()
        startPrefetching(prefetchOnThreads, prefetchMaxSize, 1, 0, steps, 0, buffer, generator)
        var bufferSizeSum = 0
        var lastReportStep = 0
        val stepsMetrics = mutableMapOf<String, MutableList<Float>>()

        for (step in 0 until steps) {
            while (buffer.isEmpty())
                Thread.sleep(100)

            val (x, y) = buffer.pop()

            val (xTransferred, yTransferred) = transferXYToModelPaltform(x, y)

            val predictions = predictOnBatchToTargetPlatform(xTransferred, this.platform)

            metrics.forEach { outputIndex, namedMetric ->
                stepsMetrics.putIfAbsent(namedMetric.first, mutableListOf())
                stepsMetrics[namedMetric.first]?.add(namedMetric.second.calculate(predictions[outputIndex], yTransferred[outputIndex]!!))
            }

            if (verbose) {
                bufferSizeSum += buffer.size

                val stepsBetweenReport = floor(steps / 100.0).toInt() + 1

                if (step % stepsBetweenReport == 0) {
                    val bufferUtilization = "%.0f".format(bufferSizeSum.toDouble() / stepsBetweenReport / (prefetchMaxSize + 1) * 100.0)
                    bufferSizeSum = 0
                    val metrics = stepsMetrics.map { (name, values) ->
                        name to "%.4g".format(values.subList(lastReportStep, step + 1).sum() / (step + 1 - lastReportStep))
                    }.toMap()

                    println("Step: [${step + 1}/$steps] Metrics: $metrics Buffer: $bufferUtilization%")

                    lastReportStep = step + 1
                }
            }
        }

        return stepsMetrics.mapValues { (_, values) -> values.sum() / values.size }
    }

    fun evaluate(
        generator: Generator,
        verbose: Boolean = true,
        prefetchOnThreads: Int = 2,
        prefetchMaxSize: Int = 100
    ): Map<String, Float> {
        return evaluate(generator.steps, verbose, prefetchOnThreads, prefetchMaxSize, generator::generate)
    }

    private fun startPrefetching(
        threads: Int,
        maxSize: Int,
        epochs: Int,
        epochStart: Int,
        stepsPerEpoch: Int,
        stepsStart: Int,
        buffer: ConcurrentLinkedDeque<Pair<List<Tensor>, List<Tensor?>>>,
        generator: (step: Int) -> Pair<List<Tensor>, List<Tensor>>,
        transferToPlatform: Boolean = false
    ) {
        val executor = Executors.newFixedThreadPool(threads, PrefetchThreadFactory())
        for (e in epochStart until epochs)
            for (step in stepsStart until stepsPerEpoch) {
                executor.execute {
                    if (transferToPlatform) {
                        val (x, y) = generator(step)
                        val transferred = transferXYToModelPaltform(x, y)
                        buffer.push(transferred)
                    } else {
                        buffer.push(generator(step))
                    }
                    while (buffer.size >= maxSize)
                        Thread.sleep(10)
                }
            }
        executor.shutdown()
    }


    class PrefetchThreadFactory: ThreadFactory {
        private val threadNumber = AtomicInteger(0)
        private val group = ThreadGroup("prefetch")

        override fun newThread(r: Runnable): Thread {
            val thread = Thread(group, r, "prefetch-pool-" + threadNumber.getAndIncrement(), 0L)
            if (thread.isDaemon)
                thread.isDaemon = false
            if (thread.priority != Thread.NORM_PRIORITY)
                thread.priority = Thread.NORM_PRIORITY
            return thread
        }

    }

    class MetricCollectingSummaryLoss(
        private val lossFunctions: List<LossFunction?>,
        private val namedLosses: Map<String, () -> Tensor>
    ) {

        val lossValues = mutableMapOf<String, Float>()

        fun calculateLoss(yPreds: List<Tensor>, yTrues: List<Tensor?>): Tensor {
            lossValues.clear()
            val calculatedLosses = lossFunctions
                .mapIndexed { index, lossFunction -> "loss_${index + 1}" to lossFunction }
                .zip(yPreds, yTrues)
                .filter { (name, loss, _, yTrue) ->
                    if (loss != null && yTrue == null)
                        throw IllegalStateException("Null true value found for non null loss: ${name}")
                    loss != null
                }
                .map { (name, loss, yPred, yTrue) -> name to loss!!.calculateLoss(yPred, yTrue!!) }
                .toMutableList()

            calculatedLosses.addAll(namedLosses.map { (name, loss) -> name to loss.invoke() })

            val sumLoss = calculatedLosses
                .onEach { (name, loss) -> lossValues[name] = loss.item() }
                .map { (_, loss) -> loss }
                .reduce(Tensor::plus)

            lossValues["loss"] = sumLoss.item()
            return sumLoss
        }

        private fun List<Pair<String, LossFunction?>>.zip(yPreds: List<Tensor>, yTrues: List<Tensor?>): List<NamedLossWithParams> {
            return this.zip(yPreds.zip(yTrues))
                .map { (namedLoss, params) -> NamedLossWithParams(namedLoss.first, namedLoss.second, params.first, params.second) }
        }

    }

    data class NamedLossWithParams(val name: String, val function: LossFunction?, val yPred: Tensor, val yTrue: Tensor?)

}