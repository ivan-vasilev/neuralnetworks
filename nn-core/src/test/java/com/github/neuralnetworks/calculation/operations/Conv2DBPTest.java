package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationLayerCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;

/**
 * Created by chass on 25.11.14.
 */
public class Conv2DBPTest extends AbstractTest {

    /**
     * Info:
     *      - disabled testing of Runtime.CPU_SEQ
     *      - comparing CPU with OpenCL results is enough
     */

    @Test
    public void testCNNBackpropagationValues() {
        Tensor seqResult = testCNNBackpropagationValues(Runtime.CPU_SEQ);
        Tensor openclResult  = testCNNBackpropagationValues(Runtime.OPENCL);

        assertTrue(isEqual(seqResult,openclResult));
    }

    private Tensor testCNNBackpropagationValues(Runtime runtime)
    {
        // prepare engine
        configureGlobalRuntimeEnvironment(runtime);

        NeuralNetworkImpl nn = NNFactory.convNN(new int[][]{{3, 3, 2}, {2, 2, 1, 1, 1, 0, 0}}, true);
        nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, null));

        Conv2DConnection c = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
        Tensor.TensorIterator it = c.getWeights().iterator();
        float x = 0.1f;
        while (it.hasNext())
        {
            c.getWeights().getElements()[it.next()] = x;
            x += 0.1f;
        }

        Conv2DConnection b = (Conv2DConnection) nn.getOutputLayer().getConnections().get(1);
        b.getWeights().getElements()[b.getWeights().getStartIndex()] = -3f;

        SimpleInputProvider ts = new SimpleInputProvider(new float[][] { { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f } }, new float[][] { {
                1, 1, 1, 1 } });
        BackPropagationTrainer<?> trainer = TrainerFactory.backPropagation(nn, ts, null, null, null, 0.5f, 0f, 0f, 0f, 0f, 1, 1, 1);
        BackPropagationLayerCalculatorImpl bplc = (BackPropagationLayerCalculatorImpl) trainer.getBPLayerCalculator();
        bplc.setSkipEndLayers(false);
        trainer.train();

        return trainer.getBackpropagation().get(nn.getInputLayer());
    }

    @Test
    public void testCNNBackpropagation() {
        Tensor seqResult = testCNNBackpropagation(Runtime.CPU_SEQ);
        Tensor openclResult  = testCNNBackpropagation(Runtime.OPENCL);

        assertTrue(isEqual(seqResult,openclResult));
    }

    private Tensor testCNNBackpropagation(Runtime runtime)
    {
        // prepare engine
        configureGlobalRuntimeEnvironment(runtime);

        NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 3, 3, 2 }, { 2, 2, 2, 1, 1, 0, 0 } }, true);
        nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, null));

        Conv2DConnection c = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);

        Tensor.TensorIterator it = c.getWeights().iterator(new int[][] { { 0, 0, 0, 0 }, { 0, 1, 1, 1 } });
        for (float i = 0.1f; it.hasNext(); i += 0.1f)
        {
            c.getWeights().getElements()[it.next()] = i;
        }

        it = c.getWeights().iterator(new int[][] { { 1, 0, 0, 0 }, { 1, 1, 1, 1 } });
        for (float i = 0.1f; it.hasNext(); i += 0.1f)
        {
            c.getWeights().getElements()[it.next()] = i;
        }

        Conv2DConnection b = (Conv2DConnection) nn.getOutputLayer().getConnections().get(1);
        b.getWeights().getElements()[b.getWeights().getStartIndex()] = -3f;
        b.getWeights().getElements()[b.getWeights().getEndIndex()] = -3f;

        SimpleInputProvider ts = new SimpleInputProvider(new float[][] { { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f } }, new float[][] { {
                1, 1, 1, 1, 1, 1, 1, 1 } });
        BackPropagationTrainer<?> t = TrainerFactory.backPropagation(nn, ts, null, null, null, 0.5f, 0f, 0f, 0f, 0f, 1, 1, 1);
        t.train();

        return c.getWeights();
    }

    @Test
    public void testCNNBackpropagation2() {
        Matrix seqResult = testCNNBackpropagation2(Runtime.CPU_SEQ);
        Matrix openclResult  = testCNNBackpropagation2(Runtime.OPENCL);

        assertTrue(isEqual(seqResult,openclResult));
    }

    private Matrix testCNNBackpropagation2(Runtime runtime)
    {
        // prepare engine
        configureGlobalRuntimeEnvironment(runtime);

        NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 2, 1, 1 }, { 1, 1, 1, 1, 2, 3 }, { 2 }, { 2 }, { 1 } }, false);
        nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, null));
        CalculationFactory.lcMaxPooling(nn);

        FullyConnected c1 = (FullyConnected) nn.getInputLayer().getConnections().get(0).getOutputLayer().getConnections().get(1).getOutputLayer().getConnections().get(1);
        Matrix cg1 = c1.getWeights();
        cg1.set(0.1f, 0, 0);
        cg1.set(0.8f, 0, 1);
        cg1.set(0.4f, 1, 0);
        cg1.set(0.6f, 1, 1);

        FullyConnected c2 = (FullyConnected) nn.getOutputLayer().getConnections().iterator().next();
        Matrix cg2 = c2.getWeights();
        cg2.set(0.3f, 0, 0);
        cg2.set(0.9f, 0, 1);

        BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, new SimpleInputProvider(new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }), new SimpleInputProvider(
                new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }), null, null, 1f, 0f, 0f, 0f, 0f, 1, 1, 1);
        bpt.train();

        return cg1;
    }

    @Test
    public void testCNNBackpropagation3() {
        Tensor seqResult = testCNNBackpropagation3(Runtime.CPU_SEQ);
        Tensor openclResult  = testCNNBackpropagation3(Runtime.OPENCL);

        assertTrue(isEqual(seqResult,openclResult));
    }

    private Tensor testCNNBackpropagation3(Runtime runtime)
    {
        // prepare engine
        configureGlobalRuntimeEnvironment(runtime);

        NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 3, 3, 2 }, { 2, 2, 1, 1, 1, 0, 0 } }, true);
        nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, null));

        Conv2DConnection c = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
        Tensor.TensorIterator it = c.getWeights().iterator();
        float x = 0.1f;
        while (it.hasNext())
        {
            c.getWeights().getElements()[it.next()] = x;
            x += 0.1f;
        }

        Conv2DConnection b = (Conv2DConnection) nn.getOutputLayer().getConnections().get(1);
        b.getWeights().getElements()[b.getWeights().getStartIndex()] = -3f;

        SimpleInputProvider ts = new SimpleInputProvider(new float[][] { { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f },
                { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f } }, new float[][] { { 1, 1, 1, 1 }, { 1, 1, 1, 1 } });
        BackPropagationTrainer<?> t = TrainerFactory.backPropagation(nn, ts, null, null, null, 0.5f, 0f, 0f, 0f, 0f, 1, 1, 1);
        t.train();

        return c.getWeights();
    }
}
