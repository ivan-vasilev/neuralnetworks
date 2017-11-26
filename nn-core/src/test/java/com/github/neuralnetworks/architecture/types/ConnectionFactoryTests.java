package com.github.neuralnetworks.architecture.types;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * Created by chass on 02.12.14.
 */
@RunWith(Parameterized.class)
public class ConnectionFactoryTests extends AbstractTest {

    public ConnectionFactoryTests(RuntimeConfiguration conf)
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
    public void testConnectionFactory()
    {
        boolean sharedMemory = Environment.getInstance().getRuntimeConfiguration().getUseWeightsSharedMemory();
        Environment.getInstance().getRuntimeConfiguration().setUseWeightsSharedMemory(true);

        ConnectionFactory f = new ConnectionFactory();
        FullyConnected fc1 = f.fullyConnected(null, null, 2, 3);
        FullyConnected fc2 = f.fullyConnected(null, null, 5, 2);

        assertTrue(fc1.getWeights().getElements() == fc2.getWeights().getElements());
        assertEquals(16, fc1.getWeights().getElements().length, 0);
        assertEquals(0, fc1.getWeights().getStartOffset(), 0);
        assertEquals(3, fc1.getWeights().getRows(), 0);
        assertEquals(2, fc1.getWeights().getColumns(), 0);
        fc1.getWeights().set(3, 1, 1);
        assertEquals(3, fc1.getWeights().get(1, 1), 0);

        assertEquals(6, fc2.getWeights().getStartOffset(), 0);
        assertEquals(2, fc2.getWeights().getRows(), 0);
        assertEquals(5, fc2.getWeights().getColumns(), 0);
        fc2.getWeights().set(5, 1, 1);
        assertEquals(5, fc2.getWeights().get(1, 1), 0);

        Conv2DConnection c = f.conv2d(null, null, 3, 3, 3, 2, 2, 3, 1, 1, 0, 0);
        assertEquals(52, c.getWeights().getElements().length, 0);
        assertEquals(36, c.getWeights().getSize(), 0);
        assertEquals(16, c.getWeights().getStartOffset(), 0);
        assertEquals(4, c.getWeights().getDimensions().length, 0);

        Environment.getInstance().getRuntimeConfiguration().setUseWeightsSharedMemory(sharedMemory);
    }

}
