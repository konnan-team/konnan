package eu.redbean.konnan.dataprocessing.batchtransform

import eu.redbean.kten.api.tensor.Tensor

fun interface BatchTransform {
    fun transform(batch: Pair<List<Tensor>, List<Tensor>>): Pair<List<Tensor>, List<Tensor>>
}