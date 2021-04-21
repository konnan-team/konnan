package eu.redbean.konnan.dataprocessing.datasources

import com.eclipsesource.json.JsonObject
import com.eclipsesource.json.JsonValue
import eu.redbean.konnan.dataprocessing.datafetchers.JsonValueDataFetcher
import java.nio.file.Files
import java.nio.file.Path
import java.util.stream.Collectors

class FolderDataSource(
    folderPath: Path,
    filter: (Path) -> Boolean = { true }
): AbstractDataSource() {

    override val size: Long
        get() = files.size.toLong()

    val files: List<String>

    init {
        files = Files.walk(folderPath)
            .filter { Files.isRegularFile(it) }
            .filter(filter)
            .map { it.toAbsolutePath().toString() }
            .collect(Collectors.toList())
            .sorted()
    }

    fun fromJson(valueExtractor: (JsonValue) -> FloatArray) = JsonValueDataFetcher(this, valueExtractor)

    override fun close() {}

}