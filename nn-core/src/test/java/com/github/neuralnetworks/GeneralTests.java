package com.github.neuralnetworks;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorTensorFunctions;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationLayerCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * Created by chass on 02.12.14.
 * tests from ivans test classes which could not be assigned to any specific test class
 */
@RunWith(Parameterized.class)
public class GeneralTests extends AbstractTest
{

	public GeneralTests(RuntimeConfiguration conf)
	{
		Environment.getInstance().setRuntimeConfiguration(conf);
	}

	@Parameterized.Parameters
	public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> configurations = new ArrayList<>();

//		RuntimeConfiguration conf1 = new RuntimeConfiguration();
//		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//		conf1.setUseDataSharedMemory(false);
//		conf1.setUseWeightsSharedMemory(false);
//		configurations.add(new RuntimeConfiguration[] { conf1 });
//
//		RuntimeConfiguration conf2 = new RuntimeConfiguration();
//		conf2.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//		conf2.setUseDataSharedMemory(true);
//		conf2.setUseWeightsSharedMemory(true);
//		configurations.add(new RuntimeConfiguration[] { conf2 });

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
	public void testDropoutConstruction()
	{
		NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 2, 2, 1 }, false);
		new NNRandomInitializer(new RandomInitializerImpl(-0.5f, 0.5f)).initialize(mlp);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 0f, 0f } }, new float[][] { { 0f } }), null, null, null, 0f, 0f, 0f, 0f, 0.5f, 1, 1,
				1);

		LayerCalculatorImpl lc = (LayerCalculatorImpl) mlp.getLayerCalculator();
		Layer hidden = mlp.getLayers().stream().filter(l -> l != mlp.getInputLayer() && l != mlp.getOutputLayer()).findFirst().get();
		assertTrue(OperationsFactory.hasDropout(lc.getConnectionCalculator(hidden)));
		FullyConnected fc = (FullyConnected) mlp.getOutputLayer().getConnections().get(0);
		float w = fc.getWeights().get(0, 0);

		BackPropagationLayerCalculatorImpl bplc = (BackPropagationLayerCalculatorImpl) bpt.getBPLayerCalculator();
		ConnectionCalculatorTensorFunctions tf = (ConnectionCalculatorTensorFunctions) bplc.getConnectionCalculator(hidden);
		assertTrue(tf.getActivationFunctions().stream().filter(f -> OperationsFactory.isMask(f)).findAny().isPresent());

		bpt.train();

		assertTrue(!OperationsFactory.hasDropout(lc.getConnectionCalculator(hidden)));
		assertEquals(w / 2, fc.getWeights().get(0, 0), 0);
	}
}
