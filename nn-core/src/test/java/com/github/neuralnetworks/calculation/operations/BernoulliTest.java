package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertFalse;

import java.util.Random;

import org.junit.Test;

import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;

/**
 * Created by akohl on 08.12.2014.
 */
public class BernoulliTest extends AbstractTest {

    @Test
    public void testBernoulliDistribution(){
//        testBernoulliDistribution(Runtime.OPENCL);
        testBernoulliDistribution(Runtime.CPU_SEQ);
    }

    private void testBernoulliDistribution(Runtime runtime)
    {
        // Bernoulli sets output to value 0.f or 1.f,
        // decision

        configureGlobalRuntimeEnvironment(runtime);

        Random r = getRandom(13265498L);

        Tensor inputOutput = TensorFactory.tensor(50, 50);
        new RandomInitializerImpl(r, -100.0f, 100.0f).initialize(inputOutput);

        TensorFunction bd = OperationsFactory.bernoulliDistribution();
        bd.value(inputOutput);

        for(int i = 0; i < inputOutput.getElements().length; ++i)
        {
            if(     inputOutput.getElements()[i] != 0.f
                ||  inputOutput.getElements()[i] != 1.f)
            {
                assertFalse(false);
            }
        }
    }
}
