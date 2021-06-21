# Konnan - Machine learning framework for Kotlin

Konnan is a machine learning framework implemented in Kotlin, using [KTen](https://github.com/konnan-team/kten).


## Build

To build the project you have to install KTen first (not in maven central yet), then build Konnan:

```
git clone https://github.com/konnan-team/kten.git
cd kten
mvn clean install -DskipTests
cd ..
git clone https://github.com/konnan-team/konnan.git
cd konnan
mvn clean install -DskipTests
```

## Usage

To use Konnan, you have to include it in your project's dependencies, and at least the default KTen backend implementation (`kten-jvm`). For example using maven:

```xml
<dependencies>
    <!-- other dependencies -->
    <dependency>
        <groupId>eu.redbean</groupId>
        <artifactId>konnan</artifactId>
        <version>0.1.0-SNAPSHOT</version>
    </dependency>
    <dependency>
        <groupId>eu.redbean.kten.backend</groupId>
        <artifactId>kten-jvm</artifactId>
        <version>0.1.0-SNAPSHOT</version>
    </dependency>
</dependencies>
```


However, it is recommended to include the OpenCL implementation as well, to be able to utilize any OpenCL compute engine such as GPUs:

```xml
<dependency>
    <groupId>eu.redbean.kten.backend</groupId>
    <artifactId>kten-opencl</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

### Building models

Once the dependencies are added, you can start building your models. Konnan defines `Layer`s to build your neural-networks, these layers can be stacked on each other, and passed to a `Model` for training and inference. Models accept loss functions, and an optimizer in preparation for training.

Example:

```kotlin
val input = Input(28 * 28)
var layer = Dense(256)(input)
layer = ReLU()(layer)
layer = Dense(128)(layer)
layer = ReLU()(layer)
layer = Dense(10)(layer)
layer = Softmax()(layer)
val model = Model(input, layer)
model.prepare(Adam(), CategoricalCrossEntropy())
```

_The above model can be applied to the famous [MNIST](http://yann.lecun.com/exdb/mnist/) dataset._

### Training models

There are a couple of ways to train your model (building minibatches manually and passing it to the model's `trainOnBatch` method in a for-loop, or creating a lambda expression that builds minibatches and passing that to the `fitGenerator`), but maybe the most convenient way is to use the `Generator` class.

The `Generator` accepts data (X) and target (Y) fetchers as a source for the batch generation. 

`DataFetcher`s provide the functionality for fetching different data formats in Tensor converted format, from a specific `DataSource`. 

`DataSource`es provide a simple way to access data, stored in different ways, and also serves as factories for the `DataFetcher`s.

For example to access the MNIST dataset training images, you could use the `UrlDataSource` like this:

```kotlin
var trainImgs = UrlDataSource("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
            .cacheDownload(Path.of("/tmp"))
            .gzipBinaryData()
            .skipNBytes(16)
            .fetchBytesAsFloats(28 * 28)
```

This will download and cache the gzip file, unzip it, skips the first 16 bytes (MNIST dataset specific, the actual images start there), and creates a `DataFetcher` that fetches 28*28 byte values converted to float values into Tensors (The images in the dataset are 28x28 grayscale images, so a single dimension Tensor with 784 elements will represent a single image.). 

The `DataFetcher`s can apply transformations to the Tensors, for example if you'd like to rescale the values in the above case to a more useful range you can do so by:

```kotlin
trainImgs = trainImgs..rescale(from = 0f..255f, to = -1f..1f)
```

Labels can be fetched similarly:

```kotlin
val EYE = Tensor.eye(10)
val trainLabels = UrlDataSource("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
    .cacheDownload(Path.of("/tmp"))
    .gzipBinaryData()
    .skipNBytes(8)
    .fetchBytesAsFloats(1)
    .transform { EYE[it[0].item().toInt()] }
```

_Currently the `CategoricalCrossEnropy` loss can only accept the labels in one-hot encoded format, this is why the transform is applied._

With the data and target fetchers ready, the `Generator` can be configured. The `Generator` can also shuffle the data, to ensure a random distribution during training for each epoch:

```kotlin
val gen = Generator(trainImgs, trainLabels, batchSize = 64).shuffle()
```

Finally the model training can be started with the generator, using the `fitGenerator` method:

```kotlin
model.fitGenerator(gen, epochs = 5, prefetchOnThreads = 4)
```

The `fitGenerator` method accepts a couple of configuration parameters, like the number of epochs to train the model, verbose reporting of the training process to the standard output, checkpoint creation options and minibatch prefetching options. (In the example above, the model will train iterating though the whole dataset 5 times, and the minibatches during training will be prefetched on 4 separate threads.)

### Running on GPU

The `Model` class has a `onPlatform()` method, which can specify which KTen platform to use during training and inference. The platforms are identified by their platform-key, the available platforms-keys are printed to the standard output, when you first start to use KTen tensors. 

For example:
```
Platform specific implementations found:
[JVM, MemLeakDetectJVM]
[OpenCL - 0 - Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz, OpenCL - 1 - Intel(R) HD Graphics 530, OpenCL - 2 - AMD Radeon Pro 455 Compute Engine]
```

Here you can see the different OpenCL platforms (if `kten-opencl` is on the classpath), with the device id and the full name of the compute engine, this identifier can be used as a platform key.

To make the selection of the platform easier from code there is a `findPlatform` and a `findAllPlatforms` method in the `PlatformProvider`, 
which can accept a lambda expression as a selector, where you can filter the available `PlatformInfo`s to find the correct platform key. 

Example:
```kotlin
model.onPlatform(PlatformProvider.findPlatform { it.deviceType == DeviceType.GPU }.platformKey)
```

_`DataFetcher`s will produce tensors on the default platform even if this is set, but the prefetched minibatches will be transferred to the specified platform._

### Saving and loading learned model parameters

The `Model` class has a `saveParams(path)` and a `loadParams(path)` method for saving and loading the trainable parameters of the model's layers.

## Contribution

The project is in its infancy at this moment, so expect bugs, and missing features, the project is started as a proof of concept, and mostly created to learn about the inner workings of such frameworks. However, maybe it can grow into something useful in time. So if you'd like to help out with any contribution, it is very much appreciated!

If you find any bugs or would like to suggest a new feature, please open an [issue](https://github.com/konnan-team/konnan/issues).

If you'd like to contribute bug-fixes, or new features, please open an issue first, if there aren't any, describing the bug or feature, and mention the issue in your PR. 

