package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.tensor.Tensor.TensorIterator;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.backpropagation.WeightUpdates;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

/**
 * Test class for Conv2DFF operations
 */
@RunWith(Parameterized.class)
public class Conv2DWeightUpdatesTest extends AbstractTest
{
	// ////////////////////////////
	// Configuration starts here //
	// ////////////////////////////

	/**
	 * set to > 0 to use as constant seed
	 */
	private static long seed = 123456789;

	/**
	 * size of the minibatch. Values [1, 256]
	 */
	private int minibatchSize = 32;

	private KernelConfiguration kernelConfiguration;

	/**
	 * This method determines the different parameters of the kernel
	 * Each line of type configurations.add(...) is one configuration
	 * You can comment and uncomment necessary/unnecessary configurations
	 */
	@Parameters
	public static Collection<KernelConfiguration[]> runtimeConfigurations()
	{
		List<KernelConfiguration[]> configurations = new ArrayList<>();

		KernelConfiguration conf1 = new KernelConfiguration();
		conf1.connection = new ConnectionFactory().conv2d(new Layer(), new Layer(), 32, 32, 32, 5, 5, 32, 1, 1, 0, 0);
		conf1.kernelRuns = 1;
		conf1.testAparapi = false;
		conf1.testCpu = true;
		conf1.testOpenCL = true;
		configurations.add(new KernelConfiguration[] { conf1 });
		;

		return configurations;
	}

	// //////////////////////////
	// Configuration ends here //
	// //////////////////////////

	private Conv2DConnection connection; // this is set automatically

	public Conv2DWeightUpdatesTest(KernelConfiguration conf)
	{
		this.connection = conf.connection;
		this.kernelConfiguration = conf;
	}

	@Test
	public void test()
	{
		// initialize connection weights and input
		Random r = new Random();
		if (seed > 0)
		{
			r.setSeed(seed);
		}

		new RandomInitializerImpl(r, -1f, 1f).initialize(connection.getWeights());

		Tensor weights = connection.getWeights();

		ValuesProvider vp = TensorFactory.tensorProvider(connection, minibatchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor input = vp.get(connection.getOutputLayer());
		input.forEach(i -> input.getElements()[i] = r.nextFloat());

		ValuesProvider activations = TensorFactory.tensorProvider(connection, minibatchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor inputActivations = activations.get(connection.getInputLayer());
		inputActivations.forEach(i -> inputActivations.getElements()[i] = r.nextFloat());
		Tensor outputActivations = activations.get(connection.getOutputLayer());
		outputActivations.forEach(i -> outputActivations.getElements()[i] = r.nextFloat());

		// setup
		List<Connections> connections = new ArrayList<>();
		connections.add(connection);

		// CPU
		Tensor cpuOutput = null;
		if (kernelConfiguration.testCpu)
		{
			RuntimeConfiguration cpuConf = new RuntimeConfiguration();
			cpuConf.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
			cpuConf.setUseDataSharedMemory(false);
			cpuConf.setUseWeightsSharedMemory(false);
			Environment.getInstance().setRuntimeConfiguration(cpuConf);

			Tensor weightUpdates = TensorFactory.tensor(connection.getWeights().getDimensions());
			Tensor weightsCopy = TensorFactory.tensor(connection.getWeights().getDimensions());
			TensorFactory.copy(weights, weightsCopy);
			connection.setWeights(weightsCopy);

			WeightUpdates wu = OperationsFactory.weightUpdates(connection, vp, activations, weightUpdates);
			wu.updateWeights(0.01f, 0.1f, 0.0001f, 0.0001f);

			cpuOutput = TensorFactory.tensor(connection.getWeights().getDimensions());
			TensorFactory.copy(connection.getWeights(), cpuOutput);
			for (int i = 0; i < kernelConfiguration.kernelRuns; i++)
			{
				wu.updateWeights(0.01f, 0.1f, 0.0001f, 0.0001f);
			}
			connection.setWeights(weights);
		}

		// OpenCL
		Tensor oclOutput = null;
		if (kernelConfiguration.testOpenCL)
		{
			try
			{
				RuntimeConfiguration oclConf = new RuntimeConfiguration();
				oclConf.setCalculationProvider(CalculationProvider.OPENCL);
				oclConf.setUseDataSharedMemory(false);
				oclConf.setUseWeightsSharedMemory(false);
				oclConf.getOpenCLConfiguration().setAggregateOperations(false);
				oclConf.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
				oclConf.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
				Environment.getInstance().setRuntimeConfiguration(oclConf);

				Tensor weightUpdates = TensorFactory.tensor(connection.getWeights().getDimensions());
				Tensor weightsCopy = TensorFactory.tensor(connection.getWeights().getDimensions());
				TensorFactory.copy(weights, weightsCopy);
				connection.setWeights(weightsCopy);

				WeightUpdates wu = OperationsFactory.weightUpdates(connection, vp, activations, weightUpdates);
				wu.updateWeights(0.01f, 0.1f, 0.0001f, 0.0001f);

				oclOutput = TensorFactory.tensor(connection.getWeights().getDimensions());
				TensorFactory.copy(weightsCopy, oclOutput);

				oclConf.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);

				for (int i = 0; i < kernelConfiguration.kernelRuns; i++)
				{
					wu.updateWeights(0.01f, 0.1f, 0.0001f, 0.0001f);
				}
			} finally
			{
				connection.setWeights(weights);
			}
		}

		// Aparapi
		Tensor aparapiOutput = null;
		if (kernelConfiguration.testAparapi)
		{
			RuntimeConfiguration aparapiConf = new RuntimeConfiguration();
			aparapiConf.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.GPU);
			aparapiConf.setUseDataSharedMemory(false);
			aparapiConf.setUseWeightsSharedMemory(false);
			Environment.getInstance().setRuntimeConfiguration(aparapiConf);

			Tensor weightUpdates = TensorFactory.tensor(connection.getWeights().getDimensions());
			Tensor weightsCopy = TensorFactory.tensor(connection.getWeights().getDimensions());
			TensorFactory.copy(weights, weightsCopy);
			connection.setWeights(weightsCopy);

			WeightUpdates wu = OperationsFactory.weightUpdates(connection, vp, activations, weightUpdates);
			wu.updateWeights(0.01f, 0.1f, 0.0001f, 0.0001f);

			aparapiOutput = TensorFactory.tensor(connection.getWeights().getDimensions());
			TensorFactory.copy(connection.getWeights(), aparapiOutput);

			for (int i = 0; i < kernelConfiguration.kernelRuns; i++)
			{
				wu.updateWeights(0.01f, 0.1f, 0.0001f, 0.0001f);
			}
			connection.setWeights(weights);
		}

		if (oclOutput != null && cpuOutput != null)
		{
			TensorIterator oclIt = oclOutput.iterator();
			TensorIterator cpuIt = cpuOutput.iterator();
			TensorIterator weightsIt = weights.iterator();
			while (oclIt.hasNext() && cpuIt.hasNext())
			{
				int cpuId = cpuIt.next();
				assertFalse(cpuOutput.getElements()[cpuId] == weights.getElements()[weightsIt.next()]);
				assertEquals(oclOutput.getElements()[oclIt.next()], cpuOutput.getElements()[cpuId], 0.0001f);
			}
		}
	}

	private static class KernelConfiguration
	{
		private Conv2DConnection connection;

		/**
		 * how many times to execute each opencl kernel. Note that the output array is not erased after each cycle. This means that, while the input is always the same, consecutive executions of the same
		 * kernels will
		 * produce different results
		 */
		private int kernelRuns;

		/**
		 * set to true to test using OpenCL
		 */
		private boolean testOpenCL = true;

		/**
		 * set to true to compare the results between OpenCL and Aparapi
		 */
		private boolean testAparapi = true;

		/**
		 * set to true to inlcude CPU testing for performance comparison
		 */
		private boolean testCpu = true;
	}
}
