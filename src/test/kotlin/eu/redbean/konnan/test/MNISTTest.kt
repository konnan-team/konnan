package eu.redbean.konnan.test

import eu.redbean.konnan.dataprocessing.Generator
import eu.redbean.konnan.dataprocessing.datasources.UrlDataSource
import eu.redbean.konnan.layers.*
import eu.redbean.konnan.layers.activations.ReLU
import eu.redbean.konnan.layers.activations.Softmax
import eu.redbean.konnan.lossfunctions.CategoricalCrossEntropy
import eu.redbean.konnan.metrics.CategoricalAccuracy
import eu.redbean.konnan.models.Model
import eu.redbean.konnan.optimizers.Adam
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.platform.DeviceType
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import ij.ImagePlus
import org.junit.jupiter.api.Test
import java.nio.file.Path
import kotlin.test.assertTrue

class MNISTTest {

    @Test
    fun test_fit_gen_with_mnist() {
        val input = Input(1, 28, 28)
        var layer = Conv2D(32, kernelSize = 4, stride = 2)(input)
        layer = ReLU()(layer)
        layer = Conv2D(64, kernelSize = 4, stride = 2, padding = 1)(layer)
        layer = ReLU()(layer)
        layer = Flatten()(layer)
        layer = Dense(256)(layer)
        layer = ReLU()(layer)
        layer = Dense(10)(layer)
        layer = Softmax()(layer)

        val model = Model(input, layer)
        val platform = PlatformProvider.findPlatform { it.deviceType == DeviceType.GPU && it hasMoreMemoryThan "1500MB" }
        if (platform hasMoreMemoryThan "8GB") {
            PlatformProvider.memoryUsageScaleHint = 0.1
        }
        model.onPlatform(platform.platformKey)
        model.addMetric(CategoricalAccuracy(), "acc")
        model.summary()
        model.prepare(Adam(0.001f), CategoricalCrossEntropy())

        val tempDirPath = Path.of(System.getProperty("java.io.tmpdir"))

        val EYE = Tensor.eye(10)

        val trainImgs = UrlDataSource("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz")
            .cacheDownload(tempDirPath)
            .gzipBinaryData()
            .skipNBytes(16)
            .fetchBytesAsFloats(28 * 28)
            .rescale(0f..255f, -1f..1f)
            .transform { it.reshape(1, 28, 28) }

        val trainLabels = UrlDataSource("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz")
            .cacheDownload(tempDirPath)
            .gzipBinaryData()
            .skipNBytes(8)
            .fetchBytesAsFloats(1)
            .transform { EYE[it[0].item().toInt()] }

        val mnistGen = Generator(trainImgs, trainLabels, batchSize = 250).shuffle()

        //model.loadParams(tempDirPath.resolve("mnist.weights"))

        model.fitGenerator(mnistGen, 5, prefetchOnThreads = 4)

        //model.saveParams(tempDirPath.resolve("mnist.weights"))


        val evalImgs = UrlDataSource("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz")
            .cacheDownload(tempDirPath)
            .gzipBinaryData()
            .skipNBytes(16)
            .fetchBytesAsFloats(28 * 28)
            .rescale(0f..255f, -1f..1f)
            .transform { it.reshape(1, 28, 28) }

        val evalLabels = UrlDataSource("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz")
            .cacheDownload(tempDirPath)
            .gzipBinaryData()
            .skipNBytes(8)
            .fetchBytesAsFloats(1)
            .transform { EYE[it[0].item().toInt()] }

        val mnistEvalGen = Generator(evalImgs, evalLabels, batchSize = 250)


        val evalAccuracy = model.evaluate(mnistEvalGen)["acc"]


        println("Avg. eval accuracy: $evalAccuracy")

        assertTrue(evalAccuracy!! > 0.6f)

        val testBatch = mnistEvalGen.shuffle().generate(1)
        var testImgs = testBatch.first[0][0..15]
        val testLabels = testBatch.second[0][0..15].argMax(axis = -1)

        val predictions = model.predictOnBatch(testImgs).argMax(axis = -1)

        val labelNames = listOf("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

        testImgs = testImgs.reshape(-1, 28, 28)

        for (i in 0 until testImgs.shape[0]) {
            val title = "Predicted label: ${labelNames[predictions[i].item().toInt()]}, Actual: ${labelNames[testLabels[i].item().toInt()]}"
            println(title)
            var img = testImgs[i].unsqueeze(0).toImage()
            img = ImagePlus(title, img.processor.resize(400))
            img.show()
            Thread.sleep(2000)
            img.close()
        }

    }

}