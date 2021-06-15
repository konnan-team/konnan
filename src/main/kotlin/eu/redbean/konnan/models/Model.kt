package eu.redbean.konnan.models

import com.eclipsesource.json.JsonObject
import eu.redbean.konnan.layers.AbstractLayerBase
import eu.redbean.konnan.layers.Input
import eu.redbean.konnan.layers.interfaces.HasCustomUpdate
import eu.redbean.konnan.layers.modellayer.SingleOutputModelLayer
import eu.redbean.konnan.optimizers.AbstractOptimizer
import eu.redbean.konnan.optimizers.utils.GradientNormalizer
import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.serialization.TensorSerializer
import eu.redbean.kten.api.tensor.serialization.serializers.BinaryTensorSerializer
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.nio.file.StandardOpenOption.*
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

open class Model(
    val inputs: List<Input>,
    outputs: List<AbstractLayerBase>
) : AbstractModel(outputs) {

    constructor(input: Input, output: AbstractLayerBase) : this(listOf(input), listOf(output))

    private val checkpointRegex = "checkpoint_([0-9]+)_([0-9]+)\\.params".toRegex()

    override val layers
        get() = outputs.flatMap(this::flattenPreviousLayers).distinct()

    var trainCallback: (x: List<Tensor>, y: List<Tensor?>) -> Unit = {x, y -> }

    var gradNormalizer: GradientNormalizer? = null

    override fun internalTrainOnBatch(x: List<Tensor>, y: List<Tensor?>): Map<String, Float> {
        if (inputs.size != x.size || outputs.size != y.size)
            throw IllegalArgumentException("Invalid number of values. Model requires ${inputs.size} input and ${outputs.size} output values, " +
                    "but got ${x.size} input and ${y.size} output values.")

        layers.forEach {
            it.clearOutput()
            it.training = true
        }

        inputs.zip(x)
            .forEach { (inputLayer, input) -> inputLayer.forward(input) }

        val lossValue = loss.calculateLoss(outputs.map { it.value }, y)
        val metricValues = calculateMetrics(y)

        val currentLoss = lossValue.item()

        lossValue.backward()

        val gradNorm = gradNormalizer
        if (gradNorm != null) {
            val optimizableParameters = layers.flatMap { it.parameters.entries }
                .filter { !it.key.startsWith("#") }
                .map { it.value }
                .filter { it is Variable }
                .map { it as Variable }
            gradNorm.normalize(optimizableParameters)
        }

        optimizer.preUpdate(currentLoss)
        updateParameters(optimizer)
        optimizer.postUpdate()

        trainCallback(x, y)
        lossValue.release()
        return loss.lossValues + metricValues
    }

    override fun internalPredictOnBatch(x: List<Tensor>): List<Tensor> {
        if (inputs.size != x.size)
            throw IllegalArgumentException("Invalid number of inputs: ${x.size} for model with input size: ${inputs.size}")

        layers.forEach {
            it.training = false
            it.clearOutput()
        }
        inputs.zip(x).forEach { (inputLayer, input) -> inputLayer.forward(input) }
        return outputs.map {
            val res = it.value.noGrad()
            it.value.release()
            res
        }
    }

    internal fun updateParameters(optimizer: AbstractOptimizer) {
        layers.filter { it !is Input }
            .forEach {
                if (it is HasCustomUpdate) {
                    it.updateParameters(optimizer)
                } else {
                    optimizer.updateParameters(it)
                }
            }
    }

    private fun flattenPreviousLayers(layer: AbstractLayerBase): List<AbstractLayerBase> {
        return layer.previousLayers.flatMap(this::flattenPreviousLayers) + layer
    }

    operator fun invoke(vararg layers: AbstractLayerBase): AbstractLayerBase {
        if (outputs.size != 1)
            throw UnsupportedOperationException("Using model with multiple outputs as layer is not supported yet")

        return SingleOutputModelLayer(this).invoke(*layers)
    }

    fun <T> serializeParameters(serializer: TensorSerializer<T>): List<Map<String, T>> {
        return layers.map { it.serializeParametersWith(serializer).toMap() }
            .toList()
    }

    fun <T> deserializeParameters(serializer: TensorSerializer<T>, layerParams: List<Map<String, T>>) {
        if (layers.size != layerParams.size)
            throw IllegalArgumentException("Invalid serialized params, model layer size and serialized params size must be the same")

        layers.zip(layerParams).forEach { (layer, paramsMap) ->
            layer.deserializeParametersWith(serializer, paramsMap.toList())
        }
    }

    fun saveParams(path: Path, withOptimizerState: Boolean = false) {
        if (Files.exists(path) && Files.isRegularFile(path).not())
            throw IllegalArgumentException("Parameter path: $path is not a file")

        val serializedParams = serializeParameters(BinaryTensorSerializer)

        BufferedOutputStream(Files.newOutputStream(path, CREATE, WRITE, TRUNCATE_EXISTING)).use {
            ZipOutputStream(it).use {
                it.putNextEntry(ZipEntry("layer_params.obj"))
                ObjectOutputStream(it).writeObject(serializedParams)

                if (withOptimizerState) {
                    it.putNextEntry(ZipEntry("optim_state.obj"))
                    ObjectOutputStream(it).writeObject(optimizer.createSerializableState())
                }
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    fun loadParams(path: Path) {
        if (Files.isRegularFile(path).not() || Files.exists(path).not())
            throw IllegalArgumentException("Parameter path: $path is not a file, or does not exists")


        BufferedInputStream(Files.newInputStream(path)).use {
            ZipInputStream(it).use {
                if (it.nextEntry?.name == "layer_params.obj") {
                    val ois = ObjectInputStream(it)
                    deserializeParameters(BinaryTensorSerializer, ois.readObject() as List<Map<String, ByteArray>>)

                }

                if (it.nextEntry?.name == "optim_state.obj") {
                    val ois = ObjectInputStream(it)
                    optimizer.setStateFromStateMap(ois.readObject() as Map<String, Number>)
                }
            }
        }
    }

    override fun createCheckpoint(checkpointsPath: Path, epoch: Int, step: Int): Path {
        val path = checkpointsPath.resolve("checkpoint_${epoch}_${step}.params")
        saveParams(path, true)
        return path
    }

    override fun tryToLoadFromCheckpoint(checkpointsPath: Path?): Pair<Int, Int> {
        return if (checkpointsPath != null) loadFromCheckpoint(checkpointsPath) else Pair(0, 0)
    }

    fun loadFromCheckpoint(checkpointsPath: Path): Pair<Int, Int> {
        if (!Files.isDirectory(checkpointsPath)) {
            throw IllegalArgumentException("Checkpoint path must be an existing directory!")
        }

        val optionalHighestCheckpointFile = Files.walk(checkpointsPath, 1)
            .filter { Files.isRegularFile(it) }
            .filter { it.fileName.toString().matches(checkpointRegex) }
            .map {
                val (_, epochStr, stepStr) = checkpointRegex.find(it.fileName.toString())!!.groupValues
                Triple(epochStr.toInt(), stepStr.toInt(), it)
            }
            .sorted { c1, c2 -> if (c1.first != c2.first) c2.first.compareTo(c1.first) else c2.second.compareTo(c1.second) }
            .findFirst()

        if (optionalHighestCheckpointFile.isPresent) {
            val fileTriple = optionalHighestCheckpointFile.get()
            loadParams(fileTriple.third)
            return Pair(fileTriple.first, fileTriple.second)
        }

        return Pair(0, 0)
    }

}