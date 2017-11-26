package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.junit.Test;

import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;

/**
 * Created by chass on 28.11.14.
 */
public class SoftmaxTest extends AbstractTest {

    @Test
    public void testSoftMax(){
        testSoftMax(Runtime.CPU_SEQ);
        testSoftMax(Runtime.OPENCL);
    }

    private void testSoftMax(Runtime runtime)
    {

        configureGlobalRuntimeEnvironment(runtime);

        Tensor m = TensorFactory.tensor(2, 3);
        m.set(1, 0, 0);
        m.set(2, 0, 1);
        m.set(3, 0, 2);
        m.set(4, 1, 0);
        m.set(5, 1, 1);
        m.set(6, 1, 2);

        float sum1 = (float) (Math.exp(m.get(0, 0)) + Math.exp(m.get(0, 1)) + Math.exp(m.get(0, 2)));
        float sum2 = (float) (Math.exp(m.get(1, 0)) + Math.exp(m.get(1, 1)) + Math.exp(m.get(1, 2)));

        TensorFunction sf = OperationsFactory.softmaxFunction();
        sf.value(m);

        assertEquals(Math.exp(1) / sum1, m.get(0, 0), 0.000001);
        assertEquals(Math.exp(2) / sum1, m.get(0, 1), 0.000001);
        assertEquals(Math.exp(3) / sum1, m.get(0, 2), 0.000001);
        assertEquals(Math.exp(4) / sum2, m.get(1, 0), 0.000001);
        assertEquals(Math.exp(5) / sum2, m.get(1, 1), 0.000001);
        assertEquals(Math.exp(6) / sum2, m.get(1, 2), 0.000001);
    }

    @Test
    public void testSoftMax2(){
        testSoftMax2(Runtime.CPU_SEQ);
        testSoftMax2(Runtime.OPENCL);
    }

    private void testSoftMax2(Runtime runtime)
    {

        configureGlobalRuntimeEnvironment(runtime);

        Tensor m = TensorFactory.tensor(2, 2);
        m.set(6.60573f, 0, 0);
        m.set(-0.56356f, 0, 1);
        m.set(6.60573f, 1, 0);
        m.set(-0.56356f, 1, 1);

        float sum = (float) (Math.exp(m.get(0, 0)) + Math.exp(m.get(0, 1)));

        TensorFunction sf = OperationsFactory.softmaxFunction();
        sf.value(m);

        assertEquals(Math.exp(6.60573f) / sum, m.get(0, 0), 0.000001);
        assertEquals(Math.exp(-0.56356f) / sum, m.get(0, 1), 0.000001);
        assertEquals(Math.exp(6.60573f) / sum, m.get(1, 0), 0.000001);
        assertEquals(Math.exp(-0.56356f) / sum, m.get(1, 1), 0.000001);
    }

    @Test
    public void testSoftMaxRandom() {
        testSoftMaxRandom(Runtime.CPU_SEQ);
        testSoftMaxRandom(Runtime.OPENCL);
    }

    private void testSoftMaxRandom(Runtime runtime) {
        configureGlobalRuntimeEnvironment(runtime);

        // set to > 0 to use as constant seed
        long seed               = 13265498L;

        // initialize connection weights and input
        Random r = new Random();
        if (seed > 0)
        {
            r.setSeed(seed);
        }

        Tensor m = TensorFactory.tensor(20, 30);
        new RandomInitializerImpl(r, -1.0f, 1.0f).initialize(m);
        Tensor mCompare = TensorFactory.tensor(20, 30);
        mCompare.forEach(i -> mCompare.getElements()[i] = m.getElements()[i]);

        Matrix io = (Matrix)mCompare;
        float[] values = io.getElements();
        int startIndex = io.getStartIndex();
        int nextRowStep = io.getRowElementsDistance();
        int nextColumnStep = io.getColumnElementsDistance();
        int columns = io.getColumns();

        TensorFunction softmaxFunction = OperationsFactory.softmaxFunction();
        softmaxFunction.value(m);

        assertTrue(OperationsFactory.isSoftmaxFunction(softmaxFunction));

        for(int i = 0; i < (values.length / columns); ++i) {
            float sum = 0;
            int start = startIndex + i * nextRowStep;
            int c = columns;

            float max = values[start];
            for (int j = 1; j < c; j++) {
                max = Math.max(max, values[start + j * nextColumnStep]);
            }

            for (int j = 0; j < c; j++) {
                sum += Math.exp(values[start + j * nextColumnStep] - max);
            }

            for (int j = 0; j < c; j++) {
                values[start + j * nextColumnStep] = (float)Math.exp(values[start + j * nextColumnStep] - max) / sum;
            }
        }

        assertEquals(m.get(0, 0), mCompare.get(0, 0), 0.00001);
        assertEquals(m.get(0, 1), mCompare.get(0, 1), 0.00001);
        assertEquals(m.get(0, 2), mCompare.get(0, 2), 0.00001);
        assertEquals(m.get(1, 0), mCompare.get(1, 0), 0.00001);
        assertEquals(m.get(1, 1), mCompare.get(1, 1), 0.00001);
        assertEquals(m.get(1, 2), mCompare.get(1, 2), 0.00001);
    }
}
