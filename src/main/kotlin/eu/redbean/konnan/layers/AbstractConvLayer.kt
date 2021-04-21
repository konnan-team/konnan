package eu.redbean.konnan.layers

import eu.redbean.kten.api.tensor.Tensor
import kotlin.math.sqrt

abstract class AbstractConvLayer(
    val size: Int,
    val kernelSize: List<Int>,
    val padding: List<Int>,
    val dilation: List<Int>,
    val stride: List<Int>,
    val outputPadding: List<Int>,
    val groups: Int,
    val useBias: Boolean,
    val channelsFirst: Boolean,
    val transposed: Boolean,
    name: String?
): Layer(name) {

    init {
        if (size < 1)
            throw IllegalArgumentException("${this::class.simpleName} layer size must be at least 1")
    }

    var weight: Tensor by parameters
    var bias: Tensor by parameters

    abstract fun checkInputShape(inShape: List<Int>)

    override fun postInvoke() {
        val inShape = previousLayers[0].shape
        checkInputShape(inShape)

        val inputChannels = if (channelsFirst) inShape.first() else inShape.last()

        if (inputChannels % groups != 0)
            throw IllegalArgumentException("Input channels must be divisible with groups, input channels: $inputChannels groups: $groups")
        if (size % groups != 0)
            throw IllegalArgumentException("Ouput channels must be divisible with groups, output channels: $size groups: $groups")

        val stdev = 1f / sqrt((inputChannels * kernelSize.fold(1, Int::times)).toFloat())

        if (transposed) {
            weight = (platform.createRandom(listOf(inputChannels, size / groups) + kernelSize) * stdev).asVariable(requiresGrad = true)
        } else {
            weight = (platform.createRandom(listOf(size, inputChannels / groups) + kernelSize) * stdev).asVariable(requiresGrad = true)
        }

        if (useBias)
            bias = (platform.createRandom(size) * stdev).asVariable(requiresGrad = true)

        calculateShape(inShape)
    }

    open fun calculateShape(inShape: List<Int>) {
        val inputSpatialDimensions = if (channelsFirst) inShape.drop(1) else inShape.dropLast(1)
        val outputSpatialDimansions = inputSpatialDimensions.mapIndexed { index, dimSize ->
            if (transposed) {
                (dimSize - 1) * stride[index] - (2 * padding[index]) + (dilation[index] * (kernelSize[index] - 1) + 1) + outputPadding[index]
            } else {
                (dimSize + 2 * padding[index] - (dilation[index] * (kernelSize[index] - 1) + 1)) / stride[index] + 1
            }
        }

        if (channelsFirst) {
            shape = listOf(size) + outputSpatialDimansions
        } else {
            shape = outputSpatialDimansions + size
        }
    }

    private fun withChannelsFirst(input: Tensor): Tensor {
        val axes = input.shape.indices.toMutableList()
        val channels = axes.removeLast()
        axes.add(1, channels)
        return input.permute(axes)
    }

    private fun withChannelsLast(output: Tensor): Tensor {
        val axes = output.shape.indices.toMutableList()
        val channels = axes.removeAt(1)
        axes.add(channels)
        return output.permute(axes)
    }

    override fun forward(input: Tensor): Tensor {
        if (channelsFirst) {
            return forwardWithChannelsFirst(input)
        }

        val channelsFirstInput = withChannelsFirst(input)
        val output = forwardWithChannelsFirst(channelsFirstInput)
        return withChannelsLast(output)
    }

    abstract fun forwardWithChannelsFirst(input: Tensor): Tensor

}