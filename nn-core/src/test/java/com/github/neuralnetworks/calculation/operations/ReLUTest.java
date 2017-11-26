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
 * Created by chass on 02.12.14.
 */
public class ReLUTest extends AbstractTest {

    @Test
    public void testReLU() {
        testReLU(Runtime.CPU_SEQ);
        testReLU(Runtime.OPENCL);
    }

    private void testReLU(Runtime runtime) {
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
        new RandomInitializerImpl(r, -1.0f, 1.0f).initialize(m);
        Tensor mOrig = TensorFactory.tensor(2, 3);
        mOrig.forEach(i -> mOrig.getElements()[i] = m.getElements()[i]);

        TensorFunction relu = OperationsFactory.reLUFunction();
        relu.value(m);

        assertTrue(OperationsFactory.isReLUFunction(relu));

        assertEquals(m.get(0, 0), Math.max(0, mOrig.get(0, 0)), 0.00001);
        assertEquals(m.get(0, 1), Math.max(0, mOrig.get(0, 1)), 0.00001);
        assertEquals(m.get(0, 2), Math.max(0, mOrig.get(0, 2)), 0.00001);
        assertEquals(m.get(1, 0), Math.max(0, mOrig.get(1, 0)), 0.00001);
        assertEquals(m.get(1, 1), Math.max(0, mOrig.get(1, 1)), 0.00001);
        assertEquals(m.get(1, 2), Math.max(0, mOrig.get(1, 2)), 0.00001);
    }

    @Test
    public void testReLUDerivative(){
        testReLUDerivative(Runtime.CPU_SEQ);
        testReLUDerivative(Runtime.OPENCL);
    }

    private void testReLUDerivative(Runtime runtime) {

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
        new RandomInitializerImpl(r, -1.0f, 1.0f).initialize(m);
        Tensor mOrig = TensorFactory.tensor(2, 3);
        mOrig.forEach(i -> mOrig.getElements()[i] = m.getElements()[i]);

        Tensor activation = TensorFactory.tensor(2, 3);
        new RandomInitializerImpl(r, -1.0f, 1.0f).initialize(activation);

        TensorFunction.TensorFunctionDerivative reLUDerivativeFunction = OperationsFactory.reLUDerivativeFunction();
        reLUDerivativeFunction.setActivations(activation);
        reLUDerivativeFunction.value(m);

        assertTrue(OperationsFactory.isReLUDerivativeFunction(reLUDerivativeFunction));

        assertEquals(m.get(0, 0), activation.get(0, 0) <= 0f ? 0f : mOrig.get(0, 0), 0.00001);
        assertEquals(m.get(0, 1), activation.get(0, 1) <= 0f ? 0f : mOrig.get(0, 1), 0.00001);
        assertEquals(m.get(0, 2), activation.get(0, 2) <= 0f ? 0f : mOrig.get(0, 2), 0.00001);
        assertEquals(m.get(1, 0), activation.get(1, 0) <= 0f ? 0f : mOrig.get(1, 0), 0.00001);
        assertEquals(m.get(1, 1), activation.get(1, 1) <= 0f ? 0f : mOrig.get(1, 1), 0.00001);
        assertEquals(m.get(1, 2), activation.get(1, 2) <= 0f ? 0f : mOrig.get(1, 2), 0.00001);
    }

    @Test
    public void testSoftReLU() {
        testSoftReLU(Runtime.CPU_SEQ);
        testSoftReLU(Runtime.OPENCL);
    }

    private void testSoftReLU(Runtime runtime) {
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
        new RandomInitializerImpl(r, -1.0f, 1.0f).initialize(m);
        Tensor mOrig = TensorFactory.tensor(2, 3);
        mOrig.forEach(i -> mOrig.getElements()[i] = m.getElements()[i]);

        TensorFunction softReLU = OperationsFactory.softReLUFunction();
        softReLU.value(m);

        assertTrue(OperationsFactory.isSoftReLUFunction(softReLU));

        assertEquals(m.get(0, 0), Math.log(1 + Math.exp(mOrig.get(0, 0))), 0.00001);
        assertEquals(m.get(0, 1), Math.log(1 + Math.exp(mOrig.get(0, 1))), 0.00001);
        assertEquals(m.get(0, 2), Math.log(1 + Math.exp(mOrig.get(0, 2))), 0.00001);
        assertEquals(m.get(1, 0), Math.log(1 + Math.exp(mOrig.get(1, 0))), 0.00001);
        assertEquals(m.get(1, 1), Math.log(1 + Math.exp(mOrig.get(1, 1))), 0.00001);
        assertEquals(m.get(1, 2), Math.log(1 + Math.exp(mOrig.get(1, 2))), 0.00001);
    }

    @Test
    public void testSoftReLUDerivative(){
        testSoftReLUDerivative(Runtime.CPU_SEQ);
        testSoftReLUDerivative(Runtime.OPENCL);
    }

    private void testSoftReLUDerivative(Runtime runtime) {

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
        new RandomInitializerImpl(r, -1.0f, 1.0f).initialize(m);
        Tensor mOrig = TensorFactory.tensor(2, 3);
        mOrig.forEach(i -> mOrig.getElements()[i] = m.getElements()[i]);

        Tensor activation = TensorFactory.tensor(2, 3);
        new RandomInitializerImpl(r, -1.0f, 1.0f).initialize(activation);

        TensorFunction.TensorFunctionDerivative reLUDerivativeFunction = OperationsFactory.softReLUDerivativeFunction();
        reLUDerivativeFunction.setActivations(activation);
        reLUDerivativeFunction.value(m);

        assertTrue(OperationsFactory.isSoftReLUDerivativeFunction(reLUDerivativeFunction));

        assertEquals(m.get(0, 0), mOrig.get(0, 0) * (1 / (1 + Math.exp(-activation.get(0, 0)))), 0.00001);
        assertEquals(m.get(0, 1), mOrig.get(0, 1) * (1 / (1 + Math.exp(-activation.get(0, 1)))), 0.00001);
        assertEquals(m.get(0, 2), mOrig.get(0, 2) * (1 / (1 + Math.exp(-activation.get(0, 2)))), 0.00001);
        assertEquals(m.get(1, 0), mOrig.get(1, 0) * (1 / (1 + Math.exp(-activation.get(1, 0)))), 0.00001);
        assertEquals(m.get(1, 1), mOrig.get(1, 1) * (1 / (1 + Math.exp(-activation.get(1, 1)))), 0.00001);
        assertEquals(m.get(1, 2), mOrig.get(1, 2) * (1 / (1 + Math.exp(-activation.get(1, 2)))), 0.00001);
    }
}
