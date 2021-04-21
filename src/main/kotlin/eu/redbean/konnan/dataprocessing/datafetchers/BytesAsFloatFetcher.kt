package eu.redbean.konnan.dataprocessing.datafetchers

import eu.redbean.konnan.dataprocessing.datasources.InputStreamDataSource
import eu.redbean.kten.api.tensor.Tensor

class BytesAsFloatFetcher(
    itemSize: Int,
    dataSource: InputStreamDataSource,
    shape: List<Int> = listOf(itemSize)
): BinaryDataFetcher(itemSize, dataSource, { bytes -> Tensor(shape) { (0xff and bytes[it].toInt()).toFloat() } }) {
}