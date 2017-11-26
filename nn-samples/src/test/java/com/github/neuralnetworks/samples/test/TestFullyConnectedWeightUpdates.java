package com.github.neuralnetworks.samples.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.tensor.Tensor.TensorIterator;
import com.github.neuralnetworks.training.backpropagation.WeightUpdates;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

/**
 * Test class for Conv2DFF operations
 */
@Deprecated // test moved to OpenCLFullyConnectedWeightUpdatesTest
@Ignore
@RunWith(Parameterized.class)
public class TestFullyConnectedWeightUpdates
{
	// ////////////////////////////
	// Configuration starts here //
	// ////////////////////////////

	/**
	 * set to > 0 to use as constant seed
	 */
	private static long seed = -1;

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
		conf1.connection = new ConnectionFactory().fullyConnected(new Layer(), new Layer(), 784, 10);
		conf1.kernelRuns = 1;
		conf1.testAparapi = false;
		configurations.add(new KernelConfiguration[] {conf1});;

		return configurations;
	}

	// //////////////////////////
	// Configuration ends here //
	// //////////////////////////

	private FullyConnected connection; // this is set automatically

	public TestFullyConnectedWeightUpdates(KernelConfiguration conf)
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

		Matrix weights = connection.getWeights();

		ValuesProvider vp = TensorFactory.tensorProvider(connection, minibatchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Matrix input = vp.get(connection.getOutputLayer());
		input.forEach(i -> input.getElements()[i] = r.nextFloat());

		ValuesProvider activations = TensorFactory.tensorProvider(connection, minibatchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Matrix inputActivations = activations.get(connection.getInputLayer());
		inputActivations.forEach(i -> inputActivations.getElements()[i] = r.nextFloat());
		Matrix outputActivations = activations.get(connection.getOutputLayer());
		outputActivations.forEach(i -> outputActivations.getElements()[i] = r.nextFloat());

		// setup
		List<Connections> connections = new ArrayList<>();
		connections.add(connection);

		System.out.println("START KERNEL CONFIGURATION");

		// OpenCL
		Matrix oclOutput = null;
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

				Matrix weightUpdates = TensorFactory.tensor(connection.getWeights().getDimensions());
				Matrix weightsCopy = TensorFactory.tensor(connection.getWeights().getDimensions());
				TensorFactory.copy(weights, weightsCopy);
				connection.setWeights(weightsCopy);

				WeightUpdates wu = OperationsFactory.weightUpdates(connection, vp, activations, weightUpdates);
				wu.updateWeights(0.01f, 0.1f, 0.0001f, 0.0001f);

				oclOutput = TensorFactory.tensor(connection.getWeights().getDimensions());
				TensorFactory.copy(weightsCopy, oclOutput);

				oclConf.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);

				// perform "cycles" with the opencl calculator
				long start = System.currentTimeMillis();
				for (int i = 0; i < kernelConfiguration.kernelRuns; i++)
				{
					wu.updateWeights(0.01f, 0.1f, 0.0001f, 0.0001f);
				}
				long time = System.currentTimeMillis() - start;

				System.out.println("OpenCL : " + time + " ms (" + (time / 1000) + " s) for " + kernelConfiguration.kernelRuns + " kernel runs, " + ((time * 1000) / kernelConfiguration.kernelRuns)
						+ " micro seconds per kernel run");
			} finally
			{
				OpenCLCore.getInstance().finalizeDeviceAll();
				connection.setWeights(weights);
			}
		}

		// CPU
		Matrix cpuOutput = null;
		if (kernelConfiguration.testCpu)
		{
			RuntimeConfiguration cpuConf = new RuntimeConfiguration();
			cpuConf.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
			cpuConf.setUseDataSharedMemory(false);
			cpuConf.setUseWeightsSharedMemory(false);
			Environment.getInstance().setRuntimeConfiguration(cpuConf);

			Matrix weightUpdates = TensorFactory.tensor(connection.getWeights().getDimensions());
			Matrix weightsCopy = TensorFactory.tensor(connection.getWeights().getDimensions());
			TensorFactory.copy(weights, weightsCopy);
			connection.setWeights(weightsCopy);

			WeightUpdates wu = OperationsFactory.weightUpdates(connection, vp, activations, weightUpdates);
			wu.updateWeights(0.01f, 0.1f, 0.0001f, 0.0001f);

			cpuOutput = TensorFactory.tensor(connection.getWeights().getDimensions());
			TensorFactory.copy(connection.getWeights(), cpuOutput);

			// to file
			try
			{
				PrintWriter activationsPrinter = new PrintWriter("E:\\activations.txt");
				activationsPrinter.print(inputActivations.getElements()[0]);
				for (int i = 1; i < inputActivations.getElements().length; i++) {
					activationsPrinter.print(",");
					activationsPrinter.print(inputActivations.getElements()[i]);
				}
				activationsPrinter.close();

				PrintWriter weightsStartPrinter = new PrintWriter("E:\\weights.txt");
				weightsStartPrinter.print(weights.getElements()[0]);
				for (int i = 1; i < weights.getElements().length; i++) {
					weightsStartPrinter.print(",");
					weightsStartPrinter.print(weights.getElements()[i]);
				}
				weightsStartPrinter.close();
				
				PrintWriter weightsUpdatePrinter = new PrintWriter("E:\\weightsAfterUpdate.txt");
				weightsUpdatePrinter.print(cpuOutput.getElements()[0]);
				for (int i = 1; i < cpuOutput.getElements().length; i++) {
					weightsUpdatePrinter.print(",");
					weightsUpdatePrinter.print(cpuOutput.getElements()[i]);
				}
				weightsUpdatePrinter.close();

				Matrix gradient = vp.get(connection.getOutputLayer());
				PrintWriter output = new PrintWriter("E:\\output.txt");
				output.print(gradient.getElements()[0]);
				for (int i = 1; i < gradient.getElements().length; i++) {
					output.print(",");
					output.print(gradient.getElements()[i]);
				}
				output.close();

				PrintWriter parameters = new PrintWriter("E:\\parameters.txt");
				parameters.println(OpenCLCore.getKernelOptionsString((Kernel) wu));
				parameters.close();
			} catch (FileNotFoundException e)
			{
				e.printStackTrace();
			}

			// measure time
			long start = System.currentTimeMillis();
			for (int i = 0; i < kernelConfiguration.kernelRuns; i++)
			{
				wu.updateWeights(0.01f, 0.1f, 0.0001f, 0.0001f);
			}
			long time = System.currentTimeMillis() - start;

			System.out.println("CPU    : " + time + " ms (" + (time / 1000) + " s) for " + kernelConfiguration.kernelRuns + " kernel runs, " + ((time * 1000) / kernelConfiguration.kernelRuns) + " micro seconds per kernel run");

			connection.setWeights(weights);
		}
		
		// Aparapi
		Matrix aparapiOutput = null;
		if (kernelConfiguration.testAparapi)
		{
			RuntimeConfiguration aparapiConf = new RuntimeConfiguration();
			aparapiConf.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.GPU);
			aparapiConf.setUseDataSharedMemory(false);
			aparapiConf.setUseWeightsSharedMemory(false);
			Environment.getInstance().setRuntimeConfiguration(aparapiConf);

			Matrix weightUpdates = TensorFactory.tensor(connection.getWeights().getDimensions());
			Matrix weightsCopy = TensorFactory.tensor(connection.getWeights().getDimensions());
			TensorFactory.copy(weights, weightsCopy);
			connection.setWeights(weightsCopy);

			WeightUpdates wu = OperationsFactory.weightUpdates(connection, vp, activations, weightUpdates);
			wu.updateWeights(0.01f, 0.1f, 0.0001f, 0.0001f);

			aparapiOutput = TensorFactory.tensor(connection.getWeights().getDimensions());
			TensorFactory.copy(connection.getWeights(), aparapiOutput);
			
			// measure time
			long start = System.currentTimeMillis();
			for (int i = 0; i < kernelConfiguration.kernelRuns; i++)
			{
				wu.updateWeights(0.01f, 0.1f, 0.0001f, 0.0001f);
			}
			long time = System.currentTimeMillis() - start;

			System.out.println("Aparapi: " + time + " ms (" + (time / 1000) + " s) for " + kernelConfiguration.kernelRuns + " kernel runs, " + ((time * 1000) / kernelConfiguration.kernelRuns) + " micro seconds per kernel run");

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

		System.out.println("END KERNEL CONFIGURATION");
		System.out.println();
	}

	private static class KernelConfiguration
	{
		private FullyConnected connection;

		/**
		 * how many times to execute each opencl kernel. Note that the output array is not erased after each cycle. This means that, while the input is always the same, consecutive executions of the same kernels will
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
