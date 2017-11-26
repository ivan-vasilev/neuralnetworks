package com.github.neuralnetworks.training;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * Created by chass on 02.12.14.
 */
@RunWith(Parameterized.class)
public class HyperparametersTest extends AbstractTest {

    public HyperparametersTest(RuntimeConfiguration conf)
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
    public void testHyperparameters() {
        Hyperparameters hp = new Hyperparameters();
        hp.setDefaultLearningRate(0.1f);

        Object o = new Object();
        hp.setLearningRate(o, 0.5f);

        assertEquals(0.1f, hp.getDefaultLearningRate(), 0);
        assertEquals(0.5f, hp.getLearningRate(o), 0);
        assertEquals(0.1f, hp.getLearningRate(new Object()), 0);
    }
}
