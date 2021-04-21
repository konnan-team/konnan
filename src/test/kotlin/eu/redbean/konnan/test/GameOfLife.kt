package eu.redbean.kten.tensor

import eu.redbean.konnan.test.toImage
import eu.redbean.kten.api.tensor.Tensor
import java.util.stream.IntStream
import kotlin.concurrent.thread

fun main() {
    var tensor = Tensor.bernoulliDistribution(listOf(3, 120, 200), 0.5f)

    val (depth, height, width) = tensor.shape

    val imgShow = tensor.toImage()
    imgShow.processor = imgShow.processor.resize(800)
    imgShow.show()

    while(true) {
        val newState = Tensor.zerosLike(tensor)
        IntStream.range(0, depth).parallel().forEach { d ->
            IntStream.range(0, height).parallel().forEach { i ->
                IntStream.range(0, width).parallel().forEach { j ->
                    val top = if (i == 0) 0 else 1
                    val bottom = if (i == height - 1) 0 else 1
                    val left = if (j == 0) 0 else 1
                    val right = if (j == width - 1) 0 else 1

                    val before = if (d == 0) 0 else 1
                    val after = if (d == depth - 1) 0 else 1

                    newState[d, i, j] = tensor[d-before..d+after, i - top..i + bottom, j - left..j + right].sum()
                }
            }
        }

        val vals = ((newState eq 5f) * tensor) + ((newState eq 5f) * tensor) + ((newState eq 4f) * (tensor eq 0f))
        //tensor +=  - ((vals eq 0f)*tensor) + vals

        tensor = vals

        thread(true) {
            val img = tensor.clamp(-1f, 2f).toImage()
            synchronized(imgShow) {
                imgShow.processor = img.processor.resize(800)
                imgShow.updateAndDraw()
            }
        }
    }
}

