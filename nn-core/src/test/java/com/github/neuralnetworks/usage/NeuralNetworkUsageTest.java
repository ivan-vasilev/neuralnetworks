package com.github.neuralnetworks.usage;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.usage.NeuralNetworkUsage;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

@RunWith(Parameterized.class)
public class NeuralNetworkUsageTest
{

	public NeuralNetworkUsageTest(RuntimeConfiguration conf)
	{
		Environment.getInstance().setRuntimeConfiguration(conf);
	}

	@Parameterized.Parameters
	public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
	{
		RuntimeConfiguration conf1 = new RuntimeConfiguration();
		conf1.setCalculationProvider(CalculationProvider.APARAPI);
		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);

		RuntimeConfiguration conf2 = new RuntimeConfiguration();
		conf2.setCalculationProvider(CalculationProvider.OPENCL);
		conf2.setUseDataSharedMemory(false);
		conf2.setUseWeightsSharedMemory(false);
		conf2.getOpenCLConfiguration().setAggregateOperations(false);
		conf2.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
		conf2.getOpenCLConfiguration().setUseOptionsString(false);
		conf2.getOpenCLConfiguration().setPushToDeviceBeforeOperation(true);
		conf2.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(true);
		conf2.getOpenCLConfiguration().setRestartLibraryAfterPhase(true);
		conf2.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);

		return Arrays.asList(new RuntimeConfiguration[][] { { conf1 }, { conf2 } });
	}

	@Test
	public void testClassifyVector() throws Exception
	{

		// create a network
		NeuralNetworkBuilder neuralNetworkBuilder = new NeuralNetworkBuilder();
		neuralNetworkBuilder.addLayerBuilder(new InputLayerBuilder("input", 2, 2, 1));
		FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(2);
		fullyConnectedLayerBuilder.setAddBias(false);
		fullyConnectedLayerBuilder.setActivationType(ActivationType.Nothing);
		neuralNetworkBuilder.addLayerBuilder(fullyConnectedLayerBuilder);

		NeuralNetworkImpl neuralNetwork = neuralNetworkBuilder.build();
		for (Layer layer : neuralNetwork.getLayers())
		{
			((FullyConnected) layer.getConnections().get(0)).getWeights().setElements(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 });
		}


		// create a test vector
		float[] testVector = new float[] { 1, 2, 3, 4 };

		// test
		NeuralNetworkUsage neuralNetworkUsage = new NeuralNetworkUsage(neuralNetwork);
		float[] target = neuralNetworkUsage.classifyVector(testVector);

		Assert.assertArrayEquals("The output of the neural network isn't correct!", new float[] { 30, 70 }, target, 0);
	}
}