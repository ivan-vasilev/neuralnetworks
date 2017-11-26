package com.github.neuralnetworks.input;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.input.SimpleFileInputProvider;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.TrainingInputDataImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.Util;

/**
 * Created by chass on 02.12.14.
 */
@RunWith(Parameterized.class)
public class SimpleFileInputProviderTest extends AbstractTest {

    public SimpleFileInputProviderTest(RuntimeConfiguration conf)
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
    public void testSimpleFileInputProvider() {
        SimpleFileInputProvider sip2 = null;
        try
        {
            SimpleInputProvider sip1 = new SimpleInputProvider(new float[][] { { 1, 2 }, { 3, 4 }, { 5, 6 } }, new float[][] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            Util.inputToFloat(sip1, "ip_input", "ip_target");
            sip1.reset();

            sip2 = new SimpleFileInputProvider("ip_input", "ip_target", sip1.getInputDimensions(), sip1.getTargetDimensions(), sip1.getInputSize());
            TrainingInputDataImpl ti1 = new TrainingInputDataImpl(TensorFactory.tensor(2, sip1.getInputDimensions()), TensorFactory.tensor(2, sip1.getTargetDimensions()));
            TrainingInputDataImpl ti2 = new TrainingInputDataImpl(TensorFactory.tensor(2, sip2.getInputDimensions()), TensorFactory.tensor(2, sip2.getTargetDimensions()));

            for (int i = 0; i < sip2.getInputSize(); i += 2)
            {
                sip1.populateNext(ti1);
                sip2.populateNext(ti2);
                assertTrue(Arrays.equals(ti1.getInput().getElements(), ti2.getInput().getElements()));
                assertTrue(Arrays.equals(ti1.getTarget().getElements(), ti2.getTarget().getElements()));
            }
        } finally
        {
            try
            {
                sip2.close();
                Files.delete(Paths.get("ip_input"));
                Files.delete(Paths.get("ip_target"));
            } catch (IOException e)
            {
                e.printStackTrace();
            }
        }
    }

}
