package eu.redbean.konnan.dataprocessing

import eu.redbean.konnan.dataprocessing.batchtransform.BatchTransform
import eu.redbean.konnan.dataprocessing.batchtransform.ShuffleBatch
import eu.redbean.konnan.dataprocessing.datafetchers.DataFetcher
import eu.redbean.konnan.dataprocessing.datasources.AbstractDataSource
import eu.redbean.kten.api.tensor.Tensor
import java.util.concurrent.ConcurrentHashMap


class Generator(
    private val dataFetchers: List<DataFetcher<*>>,
    private val targetFetchers: List<DataFetcher<*>>,
    private val batchSize: Int = 32,
    private val dropLast: Boolean = false,
    private val cacheFetched: Boolean = true
) {

    constructor(dataFetcher: DataFetcher<*>, targetFetcher: DataFetcher<*>, batchSize: Int = 32, dropLast: Boolean = false, cacheFetched: Boolean = true):
            this(listOf(dataFetcher), listOf(targetFetcher), batchSize, dropLast, cacheFetched)

    val steps: Int

    private val lastBatchSize: Int

    private val cache: MutableMap<Int, Pair<List<Tensor>, List<Tensor>>>

    private val dataSources: MutableSet<AbstractDataSource>

    private val batchTransforms = mutableListOf<BatchTransform>()

    init {
        if (dataFetchers.isEmpty())
            throw IllegalStateException("No data fetchers set")

        val fetchers = dataFetchers + targetFetchers

        val fetchersSizes = fetchers.map { it.size }.distinct()
        if (fetchersSizes.size != 1) {
            throw IllegalStateException("Data fetchers must have the same number of elements, but got fetchers with sizes: ${fetchersSizes}")
        }

        val elements = fetchersSizes[0]

        lastBatchSize = elements % batchSize
        steps = (elements / batchSize) + if (dropLast || lastBatchSize == 0) 0 else 1

        cache = ConcurrentHashMap<Int, Pair<List<Tensor>, List<Tensor>>>(steps)

        dataSources = fetchers.map { it.dataSource }.toMutableSet()
    }

    fun transform(transform: BatchTransform): Generator {
        batchTransforms += transform
        return this
    }

    fun shuffle(): Generator = transform(ShuffleBatch())

    fun generate(step: Int): Pair<List<Tensor>, List<Tensor>> {
        val fetchSize = if (dropLast.not() && step == steps - 1 && lastBatchSize != 0) lastBatchSize else batchSize
        val res = if (cacheFetched) cache.computeIfAbsent(step) { fetch(fetchSize, step) } else fetch(fetchSize, step)
        if (cache.size == steps && dataSources.isNotEmpty()) {
            synchronized(dataSources) {
                dataSources.forEach { it.close() }
                dataSources.clear()
            }
        }
        return batchTransforms.runAll(res)
    }

    private fun List<BatchTransform>.runAll(batch: Pair<List<Tensor>, List<Tensor>>): Pair<List<Tensor>, List<Tensor>> {
        val iterator = this.iterator()
        var res = if (iterator.hasNext()) iterator.next().transform(batch) else batch
        while (iterator.hasNext())
            res = iterator.next().transform(res)
        return res
    }

    private fun fetch(size: Int, step: Int): Pair<List<Tensor>, List<Tensor>> {
        return synchronized(dataSources) {
            val batch = (0 until size).map { fetchExpandSingle(step * batchSize + it) } // list of pair(dataList, targetList)

            val data = dataFetchers.indices.map { idx ->
                batch.map { it.first }
                    .map { it[idx] } // pivot batch of <data1, data2...> to <batch of data1, batch of data2...>
            }.map { Tensor.concat(it, axis = 0) } // concat to <batchTensorData1, batchTensorData2...>

            val target = targetFetchers.indices.map { idx ->
                batch.map { it.second }.map { it[idx] }
            }.map { Tensor.concat(it, axis = 0) }

            data to target
        }
    }

    private fun fetchExpandSingle(itemIndex: Int): Pair<List<Tensor>, List<Tensor>> {
        return dataFetchers.map { it.fetch(itemIndex).unsqueeze(0) } to targetFetchers.map { it.fetch(itemIndex).unsqueeze(0) }
    }


}