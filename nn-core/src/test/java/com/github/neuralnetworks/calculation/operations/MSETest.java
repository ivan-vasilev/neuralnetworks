package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.backpropagation.LossFunction;

/**
 * Created by chass on 02.12.14.
 */
public class MSETest extends AbstractTest {


    @Test
    public void testMSE(){
        testMSE(Runtime.CPU_SEQ);
        testMSE(Runtime.OPENCL);
    }

    private void testMSE(Runtime runtime)
    {

        configureGlobalRuntimeEnvironment(runtime);

        Tensor activation = TensorFactory.tensor(2, 2);
        Tensor target = TensorFactory.tensor(2, 2);
        Tensor result = TensorFactory.tensor(2, 2);

        activation.set(2f, 0, 0);
        activation.set(4f, 0, 1);
        activation.set(6f, 1, 0);
        activation.set(8f, 1, 1);

        target.set(1f, 0, 0);
        target.set(2f, 0, 1);
        target.set(3f, 1, 0);
        target.set(4f, 1, 1);

        LossFunction lf = OperationsFactory.mse();
        lf.getLossFunctionDerivative(activation, target, result);

        assertEquals(-1, result.get(0, 0), 0);
        assertEquals(-2, result.get(0, 1), 0);
        assertEquals(-3, result.get(1, 0), 0);
        assertEquals(-4, result.get(1, 1), 0);
    }

}
