package eu.redbean.konnan.dataprocessing.datafetchers

import eu.redbean.konnan.dataprocessing.datasources.AbstractDataSource
import eu.redbean.kten.api.tensor.Tensor

class TransformingDataFetcher<DS: AbstractDataSource>(
    private val base: DataFetcher<DS>,
    private val transform: (Tensor) -> Tensor
): DataFetcher<DS>(base.dataSource) {

    override val size: Int
        get() = base.size

    override fun fetch(index: Int): Tensor {
        return transform(base.fetch(index))
    }
}