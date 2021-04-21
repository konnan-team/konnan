package eu.redbean.konnan.dataprocessing.datasources

import eu.redbean.konnan.dataprocessing.datafetchers.BytesAsFloatFetcher
import java.io.InputStream

class InputStreamDataSource(
    private val baseSize: Long,
    val inputStream: InputStream
): AbstractDataSource() {

    private var skipNBytes: Long = 0

    override val size: Long
        get() = baseSize - skipNBytes

    fun skipNBytes(n: Long): InputStreamDataSource {
        skipNBytes = n
        inputStream.skipNBytes(n)
        return this
    }

    fun fetchBytesAsFloats(itemSize: Int, shape: List<Int> = listOf(itemSize)): BytesAsFloatFetcher {
        return BytesAsFloatFetcher(itemSize, this, shape)
    }

    override fun close() {
        inputStream.close()
    }

}