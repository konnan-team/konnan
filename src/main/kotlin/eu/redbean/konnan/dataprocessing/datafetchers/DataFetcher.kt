package eu.redbean.konnan.dataprocessing.datafetchers

import eu.redbean.konnan.dataprocessing.datasources.AbstractDataSource
import eu.redbean.kten.api.tensor.Tensor
import kotlin.math.abs

abstract class DataFetcher<DS : AbstractDataSource>(
    val dataSource: DS
) {

    abstract val size: Int

    abstract fun fetch(index: Int): Tensor

    fun rescale(from: ClosedRange<Float>, to: ClosedRange<Float>): TransformingDataFetcher<DS> = transform {
        (((it - from.start) / (from.endInclusive - from.start)) * (to.endInclusive - to.start)) + to.start
    }

    fun transform(transform: (Tensor) -> Tensor): TransformingDataFetcher<DS> = TransformingDataFetcher(this, transform)

}