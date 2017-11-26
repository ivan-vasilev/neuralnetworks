package com.github.neuralnetworks.input;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.input.ScalingInputFunction;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * Created by chass on 02.12.14.
 */
@RunWith(Parameterized.class)
public class ScalingInputFunctionTest extends AbstractTest {

    public ScalingInputFunctionTest(RuntimeConfiguration conf)
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
    public void testScaling()
    {
        float[][] input = new float[][] { { 1, 3 }, { -1.5f, 1.5f } };
        ScalingInputFunction si = new ScalingInputFunction();
        Matrix m = TensorFactory.matrix(input);
        si.value(m);

        assertEquals(0f, m.get(0, 0), 0);
        assertEquals(1f, m.get(0, 1), 0);
        assertEquals(0f, m.get(1, 0), 0);
        assertEquals(1f, m.get(1, 1), 0);
    }
}
