package com.github.neuralnetworks.input;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputDataImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * Created by chass on 02.12.14.
 */
@RunWith(Parameterized.class)
public class SimpleInputProviderTest extends AbstractTest {

    public SimpleInputProviderTest(RuntimeConfiguration conf)
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
    public void testSimpleInputProvider() {
        SimpleInputProvider sip = new SimpleInputProvider(new float[][] { {1, 1, 1}, {0, 0, 0} }, new float[][] { {1, 1}, {0, 0} });
        TrainingInputData ti = new TrainingInputDataImpl(TensorFactory.tensor(2, 3), TensorFactory.tensor(2, 2));
        sip.populateNext(ti);

        assertEquals(1, ti.getInput().get(0, 0), 0);
        assertEquals(1, ti.getInput().get(0, 1), 0);
        assertEquals(1, ti.getInput().get(0, 2), 0);
        assertEquals(0, ti.getInput().get(1, 0), 0);
        assertEquals(0, ti.getInput().get(1, 1), 0);
        assertEquals(0, ti.getInput().get(1, 2), 0);

        assertEquals(1, ti.getTarget().get(0, 0), 0);
        assertEquals(1, ti.getTarget().get(0, 1), 0);
        assertEquals(0, ti.getTarget().get(1, 0), 0);
        assertEquals(0, ti.getTarget().get(1, 1), 0);
    }

}
