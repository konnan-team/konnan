package eu.redbean.konnan.dataprocessing.datafetchers

import eu.redbean.konnan.dataprocessing.datasources.InputStreamDataSource
import eu.redbean.kten.api.tensor.Tensor

open class BinaryDataFetcher(
    val itemSize: Int,
    dataSource: InputStreamDataSource,
    val converter: (ByteArray) -> Tensor
): DataFetcher<InputStreamDataSource>(dataSource) {

    override val size: Int
        get() = (dataSource.size / itemSize).toInt()

    override fun fetch(index: Int): Tensor {
        val bytes = dataSource.inputStream.readNBytes(itemSize)
        if (bytes.size != itemSize)
            throw IllegalStateException("Couldn't read full item from input stream")

        return converter(bytes)
    }

}