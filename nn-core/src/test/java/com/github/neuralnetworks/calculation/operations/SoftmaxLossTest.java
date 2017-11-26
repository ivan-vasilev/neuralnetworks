package com.github.neuralnetworks.calculation.operations;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.backpropagation.LossFunction;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by chass on 02.12.14.
 */
@RunWith(Parameterized.class)
public class SoftmaxLossTest extends AbstractTest {

    public SoftmaxLossTest(RuntimeConfiguration conf)
    {
        Environment.getInstance().setRuntimeConfiguration(conf);
    }

    @Parameterized.Parameters
    public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
    {
        List<RuntimeConfiguration[]> configurations = new ArrayList<>();

		RuntimeConfiguration conf1 = new RuntimeConfiguration();
		conf1.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);
		configurations.add(new RuntimeConfiguration[] { conf1 });

		RuntimeConfiguration conf2 = new RuntimeConfiguration();
		conf2.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
		conf2.setUseDataSharedMemory(true);
		conf2.setUseWeightsSharedMemory(true);
		configurations.add(new RuntimeConfiguration[] { conf2 });

        RuntimeConfiguration conf3 = new RuntimeConfiguration();
        conf3.setCalculationProvider(RuntimeConfiguration.CalculationProvider.OPENCL);
        conf3.setUseDataSharedMemory(false);
        conf3.setUseWeightsSharedMemory(false);
        conf3.getOpenCLConfiguration().setAggregateOperations(false);
        conf3.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
        conf3.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
        configurations.add(new RuntimeConfiguration[] { conf3 });

        return configurations;
    }

    @Test
    public void testSoftmaxLoss()
    {
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

        LossFunction lf = OperationsFactory.softmaxLoss();
        lf.getLossFunctionDerivative(activation, target, result);

        assertEquals(-1, result.get(0, 0), 0);
        assertEquals(-2, result.get(0, 1), 0);
        assertEquals(-3, result.get(1, 0), 0);
        assertEquals(-4, result.get(1, 1), 0);
    }

    @Test
    public void testSoftNegativeLogProbabilty()
    {
        Tensor activation = TensorFactory.tensor(2, 2);
        activation.set(0.999230719826443f, 0, 0);
        activation.set(0.0007692801735570473f, 0, 1);
        activation.set(0.999230719826443f, 1, 0);
        activation.set(0.0007692801735570473f, 1, 1);

        Tensor target = TensorFactory.tensor(2, 2);
        target.set(1, 0, 0);
        target.set(0, 0, 1);
        target.set(1, 1, 0);
        target.set(0, 1, 1);

        LossFunction sf = OperationsFactory.softmaxLoss();
        float error = sf.getLossFunction(activation, target);

        assertEquals(0.0007695762213886436, error / 2, 0.000001);
    }

}
