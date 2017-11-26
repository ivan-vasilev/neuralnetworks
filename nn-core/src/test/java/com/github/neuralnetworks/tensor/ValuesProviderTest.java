package com.github.neuralnetworks.tensor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.Util;

/**
 * Created by chass on 02.12.14.
 */
@RunWith(Parameterized.class)
public class ValuesProviderTest extends AbstractTest {

    public ValuesProviderTest(RuntimeConfiguration conf)
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
    public void testTensorProvider()
    {
        ValuesProvider tp = new ValuesProvider(true);
        String s1 = "1";
        tp.add(s1, 2, 3);
        String s2 = "2";
        tp.add(s2, 2, 3);

        assertTrue(tp.get(s1).getElements() == tp.get(s2).getElements());
        assertEquals(12, tp.get(s1).getElements().length, 0);
        assertEquals(0, tp.get(s1).getStartIndex(), 0);
        assertEquals(12, tp.get(s2).getElements().length, 0);
        assertEquals(6, tp.get(s2).getStartIndex(), 0);

        ValuesProvider tp2 = new ValuesProvider(tp);
        tp2.add(s1, 2, 3);
        tp2.add(s2, 2, 3);

        assertTrue(tp2.get(s1).getElements() == tp.get(s2).getElements());
        assertEquals(24, tp2.get(s1).getElements().length, 0);
        assertEquals(12, tp2.get(s1).getStartIndex(), 0);
        assertEquals(18, tp2.get(s2).getStartIndex(), 0);
    }

    @Test
    public void testTensorProvider2()
    {
        boolean sharedMemory = Environment.getInstance().getRuntimeConfiguration().getUseWeightsSharedMemory();
        Environment.getInstance().getRuntimeConfiguration().setUseWeightsSharedMemory(true);

        NeuralNetworkImpl nn = new NeuralNetworkImpl();

        Layer i = new Layer();
        Layer h = new Layer();
        Layer o = new Layer();

        nn.addLayer(i);

        Environment.getInstance().getRuntimeConfiguration().setUseWeightsSharedMemory(true);
        ConnectionFactory cf = new ConnectionFactory();
        NNFactory.addFullyConnectedLayer(nn, h, cf, 2, 3, true);
        NNFactory.addFullyConnectedLayer(nn, o, cf, 4, 1, true);

        ValuesProvider tp = TensorFactory.tensorProvider(nn, 2, true);

        Matrix im = tp.get(nn.getInputLayer());
        Matrix hm1 = tp.get(h, 2, 3);
        Matrix hm2 = tp.get(h, 2, 4);

        Tensor om = tp.get(o);

        assertTrue(im == tp.get(i, 2, 2));
        assertTrue(im == tp.get(i));
        assertTrue(hm1 == tp.get(h, 2, 3));
        assertTrue(hm2 == tp.get(h, 2, 4));
        assertTrue(hm1 == TensorFactory.tensor(h, nn.getConnection(i, h), tp));
        assertTrue(hm2 == TensorFactory.tensor(h, nn.getConnection(h, o), tp));
        assertTrue(om == tp.get(o, 2, 1));
        assertTrue(om == tp.get(o));

        Environment.getInstance().getRuntimeConfiguration().setUseWeightsSharedMemory(sharedMemory);
    }

    @Test
    public void testTensorProvider3()
    {
        // simple mlp test
        Environment.getInstance().getRuntimeConfiguration().setUseWeightsSharedMemory(true);
        NeuralNetworkImpl nn = NNFactory.mlp(new int[] { 3, 4, 2 }, true);
        ValuesProvider tp = TensorFactory.tensorProvider(nn, 2, false);

        Matrix in = tp.get(nn.getInputLayer());
        Matrix hidden = tp.get(nn.getLayers().stream().filter(l -> l != nn.getInputLayer() && l != nn.getOutputLayer() && !Util.isBias(l)).findFirst().get());
        Matrix out = tp.get(nn.getOutputLayer());

        assertEquals(6, in.getElements().length, 0);
        assertEquals(2, in.getRows(), 0);
        assertEquals(3, in.getColumns(), 0);
        assertEquals(8, hidden.getElements().length, 0);
        assertEquals(2, hidden.getRows(), 0);
        assertEquals(4, hidden.getColumns(), 0);
        assertEquals(4, out.getElements().length, 0);
        assertEquals(2, out.getRows(), 0);
        assertEquals(2, out.getColumns(), 0);

        hidden.set(2, 0, 1);
        hidden.set(3, 0, 2);
        hidden.set(4, 1, 2);
        hidden.set(5, 1, 3);
        assertEquals(2, hidden.get(0, 1), 0);
        assertEquals(3, hidden.get(0, 2), 0);
        assertEquals(4, hidden.get(1, 2), 0);
        assertEquals(5, hidden.get(1, 3), 0);

        out.set(8, 0, 1);
        out.set(9, 1, 1);
        assertEquals(8, out.get(0, 1), 0);
        assertEquals(9, out.get(1, 1), 0);
    }
}
