package eu.redbean.konnan.test

import eu.redbean.kten.api.tensor.Tensor
import ij.IJ
import ij.ImagePlus
import java.util.stream.IntStream

fun Tensor.toImage(): ImagePlus {
    if (dimensions != 3 || (shape[0] != 1 && shape[0] != 3))
        throw IllegalStateException("not an image")

    var tensor = this.noGrad()

    if (tensor.shape[0] == 1)
        tensor = tensor.expand(3, tensor.shape[1], tensor.shape[2])

    val min = tensor.reshape(-1).min(0).item()
    val max = tensor.reshape(-1).max(0).item()

    tensor = ((tensor - min) / (max - min)) * 255f

    val image = IJ.createImage("", "rgb", tensor.shape[2], tensor.shape[1], 1)
    val processor = image.processor

    IntStream.range(0, tensor.shape[1]).parallel().forEach { i ->
        IntStream.range(0, tensor.shape[2]).parallel().forEach { j ->
            processor[j, i] = (tensor[0, i, j].item().toInt() shl 16) +  (tensor[1, i, j].item().toInt() shl 8) + tensor[2, i, j].item().toInt()
        }
    }

    return image
}