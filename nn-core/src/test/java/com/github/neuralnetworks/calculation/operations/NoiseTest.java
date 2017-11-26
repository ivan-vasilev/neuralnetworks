package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.junit.Ignore;
import org.junit.Test;

import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;

/**
 * Created by chass on 02.12.14.
 */
public class NoiseTest extends AbstractTest {

    @Test
    @Ignore // TODO check OpenCL implementation, it seems that it uses the 'corruptionLevel' as value (see VISION-499)
    public void testNoise(){
        testNoise(Runtime.CPU_SEQ);
        testNoise(Runtime.OPENCL);
     }

    public void testNoise(Runtime runtime)
    {
        configureGlobalRuntimeEnvironment(runtime);

        Random r = getRandom(13265498L);

        Tensor inputOutput = TensorFactory.tensor(50, 50);
        new RandomInitializerImpl(r, -100.0f, 100.0f).initialize(inputOutput);

        float corruptionLevel = 0.5f;
        float corruptedValue  = 0;

        TensorFunction noise = OperationsFactory.noise(inputOutput, corruptionLevel, corruptedValue);
        noise.value(inputOutput);

        assertTrue(OperationsFactory.isNoise(noise));

        boolean hasNoise = false;
        Tensor.TensorIterator it = inputOutput.iterator();
        while (it.hasNext() && !hasNoise)
        {
            hasNoise = inputOutput.getElements()[it.next()] == corruptedValue;
        }

        assertTrue(hasNoise);
    }
}
