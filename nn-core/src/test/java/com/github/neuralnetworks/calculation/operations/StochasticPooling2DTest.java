package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.util.Environment;

/**
 * Created by chass on 28.11.14.
 */
public class StochasticPooling2DTest extends AbstractTest {

    @Test
    public void testStochasticPooling(){
        testStochasticPooling(Runtime.CPU_SEQ);
        testStochasticPooling(Runtime.OPENCL);
    }

    private void testStochasticPooling(Runtime runtime)
    {
        configureGlobalRuntimeEnvironment(runtime);
        ConnectionFactory cf = new ConnectionFactory();

        Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 3, 3, 3, 3, 1, 3, 3, 0, 0);
        List<Connections> connections = new ArrayList<Connections>();
        connections.add(c);

        ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
        float[] src = new float[] { 1.6f, 0, 0, 0, 0, 0, 0, 0, 2.4f, 1.6f, 0, 0, 0, 0, 0, 0, 0, 2.4f };
        System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);

        ConnectionCalculator calc = OperationsFactory.stochasticPooling2D();
        calc.calculate(connections, vp, c.getOutputLayer());

        Tensor t = vp.get(c.getOutputLayer());

        assertEquals(2.08, t.get(0, 0, 0, 0), 0.01);
        assertEquals(2.08, t.get(1, 0, 0, 0), 0.01);
    }

}
