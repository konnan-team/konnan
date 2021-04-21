package eu.redbean.konnan.dataprocessing.datasources

import eu.redbean.konnan.dataprocessing.datafetchers.ImageDataFetcher
import java.nio.file.Files
import java.nio.file.Path
import java.util.stream.Collectors

class ImageFolderDataSource(
    folderPath: Path
): AbstractDataSource() {

    private val supportedFileTypes = """.*(\.jpg|\.jpeg|\.png|\.bmp|.gif)$""".toRegex()

    override val size: Long
        get() = imageFileNames.size.toLong()

    val imageFileNames: List<String>

    init {
        imageFileNames = Files.walk(folderPath)
            .filter { Files.isRegularFile(it) }
            .filter { it.fileName.toString().toLowerCase().matches(supportedFileTypes) }
            .map { it.toAbsolutePath().toString() }
            .collect(Collectors.toList())
            .sorted()
    }

    fun fetchImages() = ImageDataFetcher(this)

    override fun close() {}

}