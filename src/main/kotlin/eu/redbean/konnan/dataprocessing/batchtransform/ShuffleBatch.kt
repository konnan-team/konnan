package eu.redbean.konnan.dataprocessing.batchtransform

import eu.redbean.kten.api.tensor.Tensor

class ShuffleBatch: BatchTransform {

    override fun transform(batch: Pair<List<Tensor>, List<Tensor>>): Pair<List<Tensor>, List<Tensor>> {
        assert(batch.first.isNotEmpty() && batch.first[0].dimensions > 0 && batch.first[0].shape[0] > 0)

        val (data, target) = batch
        val batchIndices = (0 until data[0].shape[0]).shuffled()

        return data.map { arrangeBatchIndices(it, batchIndices) } to target.map { arrangeBatchIndices(it, batchIndices) }
    }

    private fun arrangeBatchIndices(tensor: Tensor, indices: List<Int>): Tensor {
        return Tensor.concat(indices.map { tensor[it].unsqueeze(0) }, axis = 0)
    }

}