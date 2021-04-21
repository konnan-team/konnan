package eu.redbean.konnan.dataprocessing.datafetchers

import com.eclipsesource.json.Json
import com.eclipsesource.json.JsonObject
import com.eclipsesource.json.JsonValue
import eu.redbean.konnan.dataprocessing.datasources.FolderDataSource
import eu.redbean.kten.api.tensor.Tensor
import java.nio.file.Files
import java.nio.file.Path

class JsonValueDataFetcher(
    dataSource: FolderDataSource,
    val valueExtract: (JsonValue) -> FloatArray
): DataFetcher<FolderDataSource>(dataSource) {

    override val size: Int
        get() = dataSource.size.toInt()

    override fun fetch(index: Int): Tensor {
        val array = valueExtract(Json.parse(Files.readString(Path.of(dataSource.files[index]))))
        return Tensor.fromArray(array)
    }
}