package eu.redbean.konnan.layers.naming

import eu.redbean.konnan.layers.AbstractLayerBase

object LayerNameProvider {

    private val classSimpleNameUid = mutableMapOf<String, Int>()
    private val explicitNames = mutableListOf<String>()

    fun calculateNameFor(layer: AbstractLayerBase, name: String?): String {
        if (name != null) {
            if (name in explicitNames)
                throw IllegalArgumentException("Layer name: $name is already used for a different layer")
            explicitNames += name
            return name
        }
        val simpleName = layer::class.simpleName!!
        var uid = classSimpleNameUid[simpleName] ?: 0
        uid++
        val calculatedName = simpleName + uid
        classSimpleNameUid[simpleName] = uid
        return calculatedName
    }

}