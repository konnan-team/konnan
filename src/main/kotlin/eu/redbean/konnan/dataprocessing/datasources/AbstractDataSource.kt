package eu.redbean.konnan.dataprocessing.datasources

import java.io.InputStream

abstract class AbstractDataSource {

    abstract val size: Long

    abstract fun close()

}