package com.github.neuralnetworks.calculation.operations;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.calculation.operations.cpu.MaxoutWinners;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.*;

import static org.junit.Assert.assertEquals;

/**
 * Created by chass on 02.12.14.
 */
@RunWith(Parameterized.class)
public class MaxoutTests extends AbstractTest {


    public MaxoutTests(RuntimeConfiguration conf)
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

//		RuntimeConfiguration conf2 = new RuntimeConfiguration();
//		conf2.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
//		conf2.setUseDataSharedMemory(true);
//		conf2.setUseWeightsSharedMemory(true);
//		configurations.add(new RuntimeConfiguration[] { conf2 });
//
//        RuntimeConfiguration conf3 = new RuntimeConfiguration();
//        conf3.setCalculationProvider(RuntimeConfiguration.CalculationProvider.OPENCL);
//        conf3.setUseDataSharedMemory(false);
//        conf3.setUseWeightsSharedMemory(false);
//        conf3.getOpenCLConfiguration().setAggregateOperations(false);
//        conf3.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
//        conf3.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
//        configurations.add(new RuntimeConfiguration[] { conf3 });

        return configurations;
    }

    /**
     * maxout ff
     */
    @Test
    @Ignore // test where disabled, find out why
    public void testMaxoutFF()
    {
        NeuralNetworkImpl nn = CalculationFactory.maxout(new int[]{2, 2}, true);

        List<Connections> c = nn.getConnections();
        FullyConnected c1 = (FullyConnected) c.get(0);
        Matrix cg1 = c1.getWeights();
        cg1.set(0.1f, 0, 0);
        cg1.set(0.5f, 0, 1);
        cg1.set(0.1f, 1, 0);
        cg1.set(0.5f, 1, 1);

        FullyConnected cb1 = (FullyConnected) c.get(1);
        Matrix cgb1 = cb1.getWeights();
        cgb1.set(0.1f, 0, 0);
        cgb1.set(0.2f, 1, 0);

        ValuesProvider results = TensorFactory.tensorProvider(nn, 2, true);
        Matrix in = results.get(nn.getInputLayer());
        in.set(8, 0, 0);
        in.set(2, 1, 0);
        in.set(1, 0, 1);
        in.set(7, 1, 1);

        Set<Layer> calculated = new HashSet<>();
        calculated.add(nn.getInputLayer());
        nn.getLayerCalculator().calculate(nn, nn.getOutputLayer(), calculated, results);

        Matrix out = results.get(nn.getOutputLayer());
        assertEquals(1.1f, out.get(0, 0), 0f);
        assertEquals(1.2f, out.get(1, 0), 0f);
        assertEquals(3.6f, out.get(0, 1), 0f);
        assertEquals(3.7f, out.get(1, 1), 0f);

        int[] winners = MaxoutWinners.getInstance().getWinners();
        int startIndex = MaxoutWinners.getInstance().getStartPositions(c.get(0));
        assertEquals(1, winners[startIndex], 0);
        assertEquals(1, winners[startIndex + 1], 0);
    }

    /**
     * maxout backpropagation
     */
    @Test
    @Ignore // test where disabled, find out why
    public void testMaxoutBP()
    {
        NeuralNetworkImpl nn = CalculationFactory.maxout(new int[] { 2, 2 }, true);

        List<Connections> c = nn.getConnections();
        FullyConnected c1 = (FullyConnected) c.get(0);
        Matrix cg1 = c1.getWeights();
        cg1.set(0.1f, 0, 0);
        cg1.set(0.5f, 0, 1);
        cg1.set(0.1f, 1, 0);
        cg1.set(0.5f, 1, 1);

        FullyConnected cb1 = (FullyConnected) c.get(1);
        Matrix cgb1 = cb1.getWeights();
        cgb1.set(0.1f, 0, 0);
        cgb1.set(0.2f, 1, 0);

        ValuesProvider results = TensorFactory.tensorProvider(nn, 2, true);
        Matrix in = results.get(nn.getInputLayer());
        in.set(8, 0, 0);
        in.set(2, 1, 0);
        in.set(1, 0, 1);
        in.set(7, 1, 1);

        BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, new SimpleInputProvider(new float[][]{{8, 2}}, new float[][]{{1}}), null, null, null, 0.9f, 0f, 0f, 0f, 0f, 1, 1, 1);
        bpt.train();

        assertEquals(0.1f, cg1.get(0, 0), 0f);
        assertEquals(0.5198f, cg1.get(0, 1), 0f);
        assertEquals(0.1f, cg1.get(1, 0), 0f);
        assertEquals(1.0184002f, cg1.get(1, 1), 0f);

        assertEquals(0.109900005f, cgb1.get(0, 0), 0f);
        assertEquals(0.45920008f, cgb1.get(1, 0), 0f);
    }

}
