package eu.redbean.konnan.dataprocessing.datafetchers

import eu.redbean.konnan.dataprocessing.datasources.ImageFolderDataSource
import eu.redbean.kten.api.tensor.Tensor
import ij.IJ
import ij.process.ImageProcessor

class ImageDataFetcher(
    dataSource: ImageFolderDataSource
): DataFetcher<ImageFolderDataSource>(dataSource) {

    override val size: Int
        get() = dataSource.imageFileNames.size

    private val transformations = mutableListOf<(ImageProcessor) -> ImageProcessor>()

    override fun fetch(index: Int): Tensor {
        val baseImage =  IJ.openImage(dataSource.imageFileNames[index]).processor.convertToRGB()

        val image = transformations.fold(baseImage) { img, transform -> transform(img) }

        val colorModel = image.colorModel
        val pixels = image.pixels as IntArray
        val decomposed = (pixels.map { colorModel.getRed(it).toFloat() }
                + pixels.map { colorModel.getGreen(it).toFloat() }
                + pixels.map { colorModel.getBlue(it).toFloat() }).toFloatArray()
        return Tensor.fromArray(decomposed, false).reshape(3, image.height, image.width)
    }

    fun transformImage(transformation: (ImageProcessor) -> ImageProcessor): ImageDataFetcher {
        transformations += transformation
        return this
    }

}