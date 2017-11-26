package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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
public class TanhTest extends AbstractTest {

    @Test
    public void testTanh(){
        testTanh(Runtime.CPU_SEQ);
        testTanh(Runtime.OPENCL);
    }

    private void testTanh(Runtime runtime) {
        configureGlobalRuntimeEnvironment(runtime);

        // set to > 0 to use as constant seed
        long seed               = 13265498L;

        // initialize connection weights and input
        Random r = new Random();
        if (seed > 0)
        {
            r.setSeed(seed);
        }

        Tensor m = TensorFactory.tensor(2, 3);
        m.forEach(i -> m.getElements()[i] = r.nextFloat());
        Tensor mOrig = TensorFactory.tensor(2, 3);
        mOrig.forEach(i -> mOrig.getElements()[i] = m.getElements()[i]);

        TensorFunction tensorFunction = OperationsFactory.tanhFunction();
        tensorFunction.value(m);

        assertTrue(OperationsFactory.isTanhFunction(tensorFunction));

        assertEquals(m.get(0, 0), Math.tan(mOrig.get(0, 0)), 0.00001);
        assertEquals(m.get(0, 1), Math.tan(mOrig.get(0, 1)), 0.00001);
        assertEquals(m.get(0, 2), Math.tan(mOrig.get(0, 2)), 0.00001);
        assertEquals(m.get(1, 0), Math.tan(mOrig.get(1, 0)), 0.00001);
        assertEquals(m.get(1, 1), Math.tan(mOrig.get(1, 1)), 0.00001);
        assertEquals(m.get(1, 2), Math.tan(mOrig.get(1, 2)), 0.00001);
    }

    @Test
    public void testTanhDerivative(){
        testTanhDerivative(Runtime.CPU_SEQ);
        testTanhDerivative(Runtime.OPENCL);
    }

    private void testTanhDerivative(Runtime runtime) {

        configureGlobalRuntimeEnvironment(runtime);

        // set to > 0 to use as constant seed
        long seed               = 13265498L;

        // initialize connection weights and input
        Random r = new Random();
        if (seed > 0)
        {
            r.setSeed(seed);
        }

        Tensor m = TensorFactory.tensor(2, 3);
        m.forEach(i -> m.getElements()[i] = r.nextFloat());
        Tensor mOrig = TensorFactory.tensor(2, 3);
        mOrig.forEach(i -> mOrig.getElements()[i] = m.getElements()[i]);

        Tensor activation = TensorFactory.tensor(2, 3);
        new RandomInitializerImpl(r, -1.0f, 1.0f).initialize(activation);

        TensorFunction.TensorFunctionDerivative tanhDerivativeFunction = OperationsFactory.tanhDerivativeFunction();
        tanhDerivativeFunction.setActivations(activation);
        tanhDerivativeFunction.value(m);

        assertTrue(OperationsFactory.isTanhDerivativeFunction(tanhDerivativeFunction));

        assertEquals(m.get(0, 0), mOrig.get(0, 0) * (1 - activation.get(0, 0) * activation.get(0, 0)), 0.00001);
        assertEquals(m.get(0, 1), mOrig.get(0, 1) * (1 - activation.get(0, 1) * activation.get(0, 1)), 0.00001);
        assertEquals(m.get(0, 2), mOrig.get(0, 2) * (1 - activation.get(0, 2) * activation.get(0, 2)), 0.00001);
        assertEquals(m.get(1, 0), mOrig.get(1, 0) * (1 - activation.get(1, 0) * activation.get(1, 0)), 0.00001);
        assertEquals(m.get(1, 1), mOrig.get(1, 1) * (1 - activation.get(1, 1) * activation.get(1, 1)), 0.00001);
        assertEquals(m.get(1, 2), mOrig.get(1, 2) * (1 - activation.get(1, 2) * activation.get(1, 2)), 0.00001);
    }
}
