package com.github.neuralnetworks.training.random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.Util;

/**
 * Created by chass on 02.12.14.
 */
@RunWith(Parameterized.class)
public class NNRandomInitializerTest extends AbstractTest {

    public NNRandomInitializerTest(RuntimeConfiguration conf)
    {
        super();
        Environment.getInstance().setRuntimeConfiguration(conf);
    }

    @Parameterized.Parameters
    public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
    {
        RuntimeConfiguration conf1 = new RuntimeConfiguration();
        conf1.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
        conf1.setUseDataSharedMemory(false);
        conf1.setUseWeightsSharedMemory(false);

        RuntimeConfiguration conf2 = new RuntimeConfiguration();
        conf2.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
        conf2.setUseDataSharedMemory(true);
        conf2.setUseWeightsSharedMemory(true);

        return Arrays.asList(new RuntimeConfiguration[][]{{conf1}, {conf2}});
    }


    @Test
    public void testRandomInitializer()
    {
        NeuralNetworkImpl nn = NNFactory.mlp(new int[]{3, 2}, true);
        NNRandomInitializer rand = new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.1f, 0.1f), 0.5f);
        rand.initialize(nn);

        for (Layer l : nn.getLayers())
        {
            if (Util.isBias(l))
            {
                Tensor t = ((FullyConnected) l.getConnections().get(0)).getWeights();
                float[] elements = t.getElements();
                t.forEach(i -> assertEquals(0.5, elements[i], 0f));
            } else
            {
                Tensor t = ((FullyConnected) l.getConnections().get(0)).getWeights();
                float[] elements = t.getElements();
                t.forEach(i -> assertTrue(elements[i] >= -0.1f && elements[i] <= 0.1f && elements[i] != 0));
            }
        }
    }

    @Test
    public void testRandomInitializer1()
    {
        NeuralNetworkImpl nn = NNFactory.mlp(new int[] { 3, 2 }, true);

        NNRandomInitializer rand = new NNRandomInitializer(new MersenneTwisterRandomInitializer(2f, 3f), new MersenneTwisterRandomInitializer(-2f, -1f));
        rand.initialize(nn);

        for (Layer l : nn.getLayers())
        {
            if (Util.isBias(l))
            {
                Tensor t = ((FullyConnected) l.getConnections().get(0)).getWeights();
                float[] elements = t.getElements();
                t.forEach(i -> assertTrue(elements[i] >= -2f && elements[i] <= -1f));
            } else
            {
                Tensor t = ((FullyConnected) l.getConnections().get(0)).getWeights();
                float[] elements = t.getElements();
                t.forEach(i -> assertTrue(elements[i] >= 2f && elements[i] <= 3f));
            }
        }
    }
}
