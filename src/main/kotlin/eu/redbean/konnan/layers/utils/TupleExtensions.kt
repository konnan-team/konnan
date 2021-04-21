package eu.redbean.konnan.layers.utils

fun tripleOf(value: Int) = Triple(value, value, value)

fun List<Int>.asPair() = Pair(this[0], this[1])

fun List<Int>.asTriple() = Triple(this[0], this[1], this[2])