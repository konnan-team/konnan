package eu.redbean.konnan.test

import eu.redbean.konnan.dataprocessing.Generator
import eu.redbean.konnan.dataprocessing.datasources.FolderDataSource
import eu.redbean.konnan.dataprocessing.datasources.ImageFolderDataSource
import eu.redbean.konnan.layers.Conv2D
import eu.redbean.konnan.layers.Dense
import eu.redbean.konnan.layers.Flatten
import eu.redbean.konnan.layers.Input
import eu.redbean.konnan.layers.activations.LeakyReLU
import eu.redbean.konnan.layers.activations.ReLU
import eu.redbean.konnan.layers.activations.Sigmoid
import eu.redbean.konnan.lossfunctions.mse
import eu.redbean.konnan.models.Model
import eu.redbean.konnan.optimizers.Adam
import eu.redbean.konnan.optimizers.schedulers.DecayOnPlateau
import eu.redbean.kten.api.tensor.platform.DeviceType
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import ij.process.ImageProcessor
import org.junit.jupiter.api.Test
import java.nio.file.Path

class FFHQFeaturesTest {

    @Test
    fun test() {
        val imgDS = ImageFolderDataSource(Path.of("/tmp/ffhq"))
        val imageIndexes = imgDS.imageFileNames.map { it.substringAfterLast("/").substringBefore(".") }

        val ageFetcher = FolderDataSource(Path.of("/tmp/ffhq-features-dataset/json")) {
            imageIndexes.contains(it.fileName.toString().substring(0 until 5))
        }
            .fromJson { jsonValue ->
                FloatArray(1) {
                    if (jsonValue.isArray && jsonValue.asArray().isEmpty.not())
                        jsonValue.asArray()[0].asObject()["faceAttributes"].asObject()["age"].asFloat()
                    else
                        0.0f
                }
            }

        val images = imgDS
            .fetchImages()
            .transformImage { it.resize(128, 128) }
            .rescale(0f..255f, -1f..1f)

        val ffhqGen = Generator(images, ageFetcher, cacheFetched = false)


        val input = Input(3, 128, 128)
        var layer = Conv2D(32, 7, 4)(input)
        layer = ReLU()(layer)
        layer = Conv2D(64, 3, 2)(layer)
        layer = ReLU()(layer)
        layer = Conv2D(128, 3, 2)(layer)
        layer = ReLU()(layer)
        layer = Conv2D(256, 3, 2)(layer)
        layer = ReLU()(layer)
        layer = Flatten()(layer)
        layer = Dense(256)(layer)
        layer = ReLU()(layer)
        layer = Dense(128)(layer)
        layer = ReLU()(layer)
        layer = Dense(1)(layer)

        val model = Model(input, layer)
        model.summary()
        model.onPlatform(PlatformProvider.findPlatform { it.deviceType == DeviceType.GPU }.platformKey)
        model.prepare(Adam(scheduler = DecayOnPlateau(0.5f, 1000, minimum = 1e-5f)), ::mse)
        model.fitGenerator(ffhqGen, 5, prefetchOnThreads = 8, prefetchMaxSize = 20)
    }

}