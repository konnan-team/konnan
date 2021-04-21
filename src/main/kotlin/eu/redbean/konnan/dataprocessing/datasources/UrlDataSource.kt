package eu.redbean.konnan.dataprocessing.datasources

import eu.redbean.konnan.dataprocessing.datasources.AbstractDataSource
import eu.redbean.konnan.dataprocessing.datasources.InputStreamDataSource
import java.io.ByteArrayInputStream
import java.io.InputStream
import java.net.URL
import java.nio.file.Files
import java.nio.file.OpenOption
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.nio.file.StandardOpenOption.*
import java.util.zip.GZIPInputStream
import java.util.zip.ZipInputStream

class UrlDataSource(
    private val url: URL
) : AbstractDataSource() {

    private val connection by lazy { url.openConnection() }
    override val size: Long
        get() = cachedSize ?: connection.contentLengthLong

    private val inputStream: InputStream
        get() = cachedInputStream ?: connection.getInputStream()

    private var cachedInputStream: InputStream? = null
    private var cachedSize: Long? = null

    constructor(url: String) : this(URL(url))

    fun binaryData(): InputStreamDataSource {
        return InputStreamDataSource(size, inputStream)
    }

    fun gzipBinaryData(): InputStreamDataSource {
        val gzip = GZIPInputStream(inputStream)
        val bytes = gzip.readAllBytes()
        return InputStreamDataSource(bytes.size.toLong(), ByteArrayInputStream(bytes))
    }

    fun cacheDownload(cacheDir: Path, verbose: Boolean = true): UrlDataSource {
        if (Files.isDirectory(cacheDir).not())
            throw IllegalArgumentException("Cache directory path isn't a directory")

        val filePath = cacheDir.resolve(Path.of(url.file).fileName)

        if (Files.exists(filePath)) {
            if (Files.isRegularFile(filePath).not())
                throw IllegalStateException("File ${filePath} already exists, but isn't a regular file")

            if (verbose)
                println("Already downloaded")

        } else {
            if (verbose)
                println("Downloading ${url.file} to ${filePath}")

            Files.newOutputStream(filePath, CREATE, TRUNCATE_EXISTING, WRITE).use {
                inputStream.copyTo(it)
            }

            if (verbose)
                println("Download finished")
        }

        cachedSize = Files.size(filePath)
        cachedInputStream = Files.newInputStream(filePath, READ)

        return this
    }

    override fun close() {
        cachedInputStream?.close()
    }

}