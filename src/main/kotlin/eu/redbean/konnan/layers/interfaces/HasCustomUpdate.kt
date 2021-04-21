package eu.redbean.konnan.layers.interfaces

import eu.redbean.konnan.optimizers.AbstractOptimizer

interface HasCustomUpdate {

    fun updateParameters(optimizer: AbstractOptimizer)

}