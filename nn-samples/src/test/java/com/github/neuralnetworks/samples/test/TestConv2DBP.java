package com.github.neuralnetworks.samples.test;

import static org.junit.Assert.assertEquals;

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

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.AparapiBackpropagationConv2D2;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.BackPropagationConv2D;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.tensor.Tensor.TensorIterator;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

/**
 * Test class for Conv2DFF operations
 */
@RunWith(Parameterized.class)
@Ignore
@Deprecated
public class TestConv2DBP
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

		// configuration 1
//		KernelConfiguration conf1 = new KernelConfiguration();
//		conf1.connection = new ConnectionFactory().conv2d(new Layer(), new Layer(), 3, 3, 2, 2, 2, 1, 1, 1, 0, 0);
//		conf1.kernelRuns = 1000;
//		configurations.add(new KernelConfiguration[] {conf1});;
		
//		KernelConfiguration conf2 = new KernelConfiguration();
//		conf2.connection = new ConnectionFactory().conv2d(new Layer(), new Layer(), 224, 224, 1, 7, 7, 96, 2, 2, 0, 0);
//		conf2.kernelRuns = 1;
//		conf2.testAparapi = false;
//		conf2.testCpu = false;
//		configurations.add(new KernelConfiguration[] {conf2});;
//
//		KernelConfiguration conf3 = new KernelConfiguration();
//		conf3.connection = new ConnectionFactory().conv2d(new Layer(), new Layer(), 224, 224, 1, 7, 7, 96, 2, 2, 0, 0);
//		conf3.kernelRuns = 1;
//		conf3.testOpenCL = false;
//		conf3.testCpu = false;
//		configurations.add(new KernelConfiguration[] {conf3});;

		KernelConfiguration conf4 = new KernelConfiguration();
		conf4.connection = new ConnectionFactory().conv2d(new Layer(), new Layer(), 224, 224, 1, 7, 7, 96, 2, 2, 0, 0);
		conf4.kernelRuns = 1;
		conf4.testOpenCL = false;
		conf4.testAparapi = false;
		configurations.add(new KernelConfiguration[] {conf4});;
		
		// configuration 3
//		KernelConfiguration conf3 = new KernelConfiguration();
//		conf3.connection = new ConnectionFactory().conv2d(new Layer(), new Layer(), 224, 224, 1, 7, 7, 96, 2, 2, 0, 0);
//		conf3.kernelRuns = 1;
//		configurations.add(new KernelConfiguration[] {conf3});;

		return configurations;
	}

	// //////////////////////////
	// Configuration ends here //
	// //////////////////////////

	private Conv2DConnection connection; // this is set automatically

	public TestConv2DBP(KernelConfiguration conf)
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

		ValuesProvider vp = TensorFactory.tensorProvider(connection, minibatchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor input = vp.get(connection.getOutputLayer());
		input.forEach(i -> input.getElements()[i] = r.nextFloat());

		ValuesProvider activations = TensorFactory.tensorProvider(connection, minibatchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor inputActivations = activations.get(connection.getInputLayer());
		inputActivations.forEach(i -> inputActivations.getElements()[i] = r.nextFloat());
		Tensor outputActivations = activations.get(connection.getOutputLayer());
		outputActivations.forEach(i -> outputActivations.getElements()[i] = r.nextFloat());

		ValuesProvider weightUpdates = new ValuesProvider(Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		weightUpdates.add(connection, connection.getWeights().getDimensions());

		// setup
		List<Connections> connections = new ArrayList<>();
		connections.add(connection);

		System.out.println("START KERNEL CONFIGURATION");

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

				Properties properties = new Properties();
				properties.setParameter(Constants.WEIGHT_UDPATES, weightUpdates);
				properties.setParameter(Constants.HYPERPARAMETERS, new Hyperparameters());
				BackPropagationConnectionCalculator oclConv = OperationsFactory.bpConnectionCalculator(this.connection, properties);
				oclConv.setActivations(activations);
				oclConv.calculate(connections, vp, connection.getInputLayer());

				oclOutput = TensorFactory.tensor(vp.get(connection.getInputLayer()).getDimensions());
				TensorFactory.copy(vp.get(connection.getInputLayer()), oclOutput);

				oclConf.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);

				// perform "cycles" with the opencl calculator
				long start = System.currentTimeMillis();
				for (int i = 0; i < kernelConfiguration.kernelRuns; i++)
				{
					oclConv.calculate(connections, vp, connection.getInputLayer());
				}
				long time = System.currentTimeMillis() - start;

				System.out.println("OpenCL : " + time + " ms (" + (time / 1000) + " s) for " + kernelConfiguration.kernelRuns + " kernel runs, " + ((time * 1000) / kernelConfiguration.kernelRuns)
						+ " micro seconds per kernel run");
			} finally
			{
				OpenCLCore.getInstance().finalizeDeviceAll();
			}
		}

		// CPU
		Tensor cpuOutput = null;
		if (kernelConfiguration.testCpu)
		{
			RuntimeConfiguration cpuConf = new RuntimeConfiguration();
			cpuConf.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
			cpuConf.setUseDataSharedMemory(false);
			cpuConf.setUseWeightsSharedMemory(false);
			Environment.getInstance().setRuntimeConfiguration(cpuConf);

			Properties properties = new Properties();
			properties.setParameter(Constants.WEIGHT_UDPATES, weightUpdates);
			properties.setParameter(Constants.HYPERPARAMETERS, new Hyperparameters());
			BackPropagationConnectionCalculator cpuConv = OperationsFactory.bpConnectionCalculator(this.connection, properties);
			cpuConv.setActivations(activations);

			ValuesProvider cpuVP = TensorFactory.tensorProvider(connection, minibatchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
			TensorFactory.copy(input, cpuVP.get(connection.getOutputLayer()));

			// prepare the kernel
			cpuConv.calculate(connections, cpuVP, connection.getInputLayer());
			cpuOutput = TensorFactory.tensor(cpuVP.get(connection.getInputLayer()).getDimensions());
			TensorFactory.copy(cpuVP.get(connection.getInputLayer()), cpuOutput);

			// to file
			try
			{
				PrintWriter in = new PrintWriter("E:\\input.txt");
				in.print(input.getElements()[0]);
				for (int i = 1; i < input.getElements().length; i++) {
					in.print(",");
					in.print(input.getElements()[i]);
				}
				in.close();

				PrintWriter weights = new PrintWriter("E:\\weights.txt");
				weights.print(connection.getWeights().getElements()[0]);
				for (int i = 1; i < connection.getWeights().getElements().length; i++) {
					weights.print(",");
					weights.print(connection.getWeights().getElements()[i]);
				}
				weights.close();

				PrintWriter output = new PrintWriter("E:\\output.txt");
				output.print(cpuOutput.getElements()[0]);
				for (int i = 1; i < cpuOutput.getElements().length; i++) {
					output.print(",");
					output.print(cpuOutput.getElements()[i]);
				}
				output.close();

				PrintWriter parameters = new PrintWriter("E:\\parameters.txt");
				BackPropagationConv2D cpucc = (BackPropagationConv2D) cpuConv;
				AparapiBackpropagationConv2D2 cc = (AparapiBackpropagationConv2D2) cpucc.getInputFunctions().get(0);
				parameters.println(OpenCLCore.getKernelOptionsString(cc));
				parameters.close();
			} catch (FileNotFoundException e)
			{
				e.printStackTrace();
			}

			// measure time
			long start = System.currentTimeMillis();
			for (int i = 0; i < kernelConfiguration.kernelRuns; i++)
			{
				cpuConv.calculate(connections, cpuVP, connection.getOutputLayer());
			}
			long time = System.currentTimeMillis() - start;

			System.out.println("CPU    : " + time + " ms (" + (time / 1000) + " s) for " + kernelConfiguration.kernelRuns + " kernel runs, " + ((time * 1000) / kernelConfiguration.kernelRuns) + " micro seconds per kernel run");
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

			Properties properties = new Properties();
			properties.setParameter(Constants.WEIGHT_UDPATES, weightUpdates);
			properties.setParameter(Constants.HYPERPARAMETERS, new Hyperparameters());
			BackPropagationConnectionCalculator aparapiConv = OperationsFactory.bpConnectionCalculator(this.connection, properties);
			aparapiConv.setActivations(activations);
			
			ValuesProvider aparapiVP = TensorFactory.tensorProvider(connection, minibatchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
			TensorFactory.copy(input, aparapiVP.get(connection.getOutputLayer()));
			
			// prepare the kernel
			aparapiConv.calculate(connections, aparapiVP, connection.getInputLayer());
			aparapiOutput = TensorFactory.tensor(aparapiVP.get(connection.getInputLayer()).getDimensions());
			TensorFactory.copy(aparapiVP.get(connection.getInputLayer()), aparapiOutput);
			
			// measure time
			long start = System.currentTimeMillis();
			for (int i = 0; i < kernelConfiguration.kernelRuns; i++)
			{
				aparapiConv.calculate(connections, aparapiVP, connection.getInputLayer());
			}
			long time = System.currentTimeMillis() - start;

			System.out.println("Aparapi: " + time + " ms (" + (time / 1000) + " s) for " + kernelConfiguration.kernelRuns + " kernel runs, " + ((time * 1000) / kernelConfiguration.kernelRuns) + " micro seconds per kernel run");
		}

		if (oclOutput != null && cpuOutput != null)
		{
			TensorIterator oclIt = oclOutput.iterator();
			TensorIterator cpuIt = cpuOutput.iterator();
			while (oclIt.hasNext() && cpuIt.hasNext())
			{
				assertEquals(oclOutput.getElements()[oclIt.next()], cpuOutput.getElements()[cpuIt.next()], 0.0001f);
			}
		}

		System.out.println("END KERNEL CONFIGURATION");
		System.out.println();
	}

	private static class KernelConfiguration
	{
		private Conv2DConnection connection;

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
