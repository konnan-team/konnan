package eu.redbean.konnan.test

import eu.redbean.konnan.layers.Dense
import eu.redbean.konnan.layers.Input
import eu.redbean.konnan.layers.activations.LeakyReLU
import eu.redbean.konnan.models.Model
import eu.redbean.konnan.optimizers.Adam
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.Tensor.Companion.sum
import eu.redbean.kten.api.tensor.Tensor.Companion.tensorOf
import eu.redbean.kten.tensor.tests.assertTensorEquals
import org.junit.jupiter.api.Test
import kotlin.math.PI

class BasicModelTest {

    @Test
    fun should_train_model_to_simple_function() {
        val input = Input(3)
        var layer = Dense(32)(input)
        layer = LeakyReLU()(layer)
        layer = Dense(1)(layer)

        val model = Model(input, layer)

        model.prepare(Adam(), { yPred, yTrue -> sum((yPred - yTrue) pow 2) })

        val x = Tensor.arrange(PI.toFloat(), (PI / 1000).toFloat(), -PI.toFloat())
        val y = Tensor.sin(x).unsqueeze(-1)
        val xTrain = x.unsqueeze(-1) pow tensorOf(1, 2, 3)

        for (e in 0 until 100) {
            var sumLoss = 0f
            for (i in 0 until 2000) {
                val metrics = model.trainOnBatch(xTrain[i..i], y[i..i])
                sumLoss += metrics["loss"]!!
            }
            if (e % 10 == 0)
                println("Loss: ${sumLoss / 2000}")
        }

        val xTest = xTrain[100..100]
        val yTest = y[100..100]

        val yPred = model.predictOnBatch(xTest)

        assertTensorEquals(yTest, yPred, 1e-5f)
    }

}
