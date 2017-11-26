package com.github.neuralnetworks.analyzing.network.comparison;

import java.io.File;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.layer.ConvolutionalLayerBuilder;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.structure.LayerBuilder;
import com.github.neuralnetworks.input.RandomInputProvider;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * @author tmey
 */
public class CPUAndGPUComparison
{
	private static final Logger logger = LoggerFactory.getLogger(CPUAndGPUComparison.class);

	private EnvironmentConfigurationComparison comparison;

	private RuntimeConfiguration confJava;
	private RuntimeConfiguration confGPU;

	public CPUAndGPUComparison()
	{
		comparison = new EnvironmentConfigurationComparison();

		confJava = new RuntimeConfiguration();
		confJava.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
		confJava.setUseDataSharedMemory(false);
		confJava.setUseWeightsSharedMemory(false);

		// confGPU (for tests)
		confGPU = new RuntimeConfiguration();
		confGPU.setCalculationProvider(RuntimeConfiguration.CalculationProvider.OPENCL);
		confGPU.setUseDataSharedMemory(false);
		confGPU.setUseWeightsSharedMemory(false);
		confGPU.getOpenCLConfiguration().setAggregateOperations(true);
		confGPU.getOpenCLConfiguration().setUseOptionsString(false);
		confGPU.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
		confGPU.getOpenCLConfiguration().setPushToDeviceBeforeOperation(false);
		confGPU.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(false);

	}

	public void compare(NeuralNetworkBuilder builder)
			throws DifferentNetworksException
	{
		File problemFileForVadim = comparison.getSimilarNetworkWeightsComparison().getProblemFilesDirForVadim();
		comparison.getSimilarNetworkWeightsComparison().setProblemFilesDirForVadim(null);

		DifferentNetworksException fullNetworkException = null;

		// first test
		try
		{
			comparison.compareNetworkWithDiffEnvConf(builder, confJava, confGPU);
		} catch (DifferentNetworksException e)
		{
			fullNetworkException = e;
		}

		if (fullNetworkException == null)
		{
			comparison.getSimilarNetworkWeightsComparison().setProblemFilesDirForVadim(problemFileForVadim);
			return;
		}

		// if problem remove bias
		logger.info("deactivate bias layers");
		builder.setOverrideAddBiasTo(false);

		DifferentNetworksException withoutBiasException = null;
		try
		{
			comparison.compareNetworkWithDiffEnvConf(builder, confJava, confGPU);
		} catch (DifferentNetworksException e)
		{
			withoutBiasException = e;

		}

		if (withoutBiasException == null)
		{
			comparison.getSimilarNetworkWeightsComparison().setProblemFilesDirForVadim(problemFileForVadim);
			throw new DifferentNetworksException("There is a problem with the bias layers!", fullNetworkException);
		}


		// if problem remove activation
		logger.info("remove activation functions");

		for (LayerBuilder layerBuilder : builder.getListOfLayerBuilder())
		{
			if (layerBuilder instanceof ConvolutionalLayerBuilder)
			{
				((ConvolutionalLayerBuilder) layerBuilder).setActivationType(ActivationType.Nothing);
			}
			if (layerBuilder instanceof ConvolutionalLayerBuilder)
			{
				((ConvolutionalLayerBuilder) layerBuilder).setActivationType(ActivationType.Nothing);
			}
		}


		comparison.getSimilarNetworkWeightsComparison().setProblemFilesDirForVadim(problemFileForVadim);

		DifferentNetworksException withoutBiasAndActivationException = null;
		try
		{
			comparison.compareNetworkWithDiffEnvConf(builder, confJava, confGPU);
		} catch (DifferentNetworksException e)
		{
			withoutBiasAndActivationException = e;

		}

		if (withoutBiasAndActivationException == null)
		{
			throw new DifferentNetworksException("There is a problem with the activation function!", withoutBiasException);
		}


		// if still a problem throw exception

		throw new DifferentNetworksException("There is a problem with the main kernels!", withoutBiasAndActivationException);

	}

	public EnvironmentConfigurationComparison getComparison()
	{
		return comparison;
	}

	public void setComparison(EnvironmentConfigurationComparison comparison)
	{
		if (comparison == null)
		{
			throw new IllegalArgumentException("comparison must be not null!");
		}

		this.comparison = comparison;
	}

	public static void main(String[] args) throws DifferentNetworksException
	{

		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();
		builder.setOverrideAddBiasTo(true);
		builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 20, 1, 1));
		builder.addLayerBuilder(new FullyConnectedLayerBuilder(20).setActivationType(ActivationType.ReLU));
		builder.addLayerBuilder(new FullyConnectedLayerBuilder(10).setActivationType(ActivationType.SoftMax));
		builder.setTrainingSet(new RandomInputProvider(3, 20, 10, new Random(123)));

		builder.setLearningRate(0.1f);
		builder.setMomentum(0.9f);
		builder.setL1weightDecay(0.01f);
		builder.setL2weightDecay(0.01f);
		builder.setEpochs(1);
		builder.setTrainingBatchSize(3);

		builder.setRand(new NNRandomInitializer(new RandomInitializerImpl(-0.01f, 0.01f)));


		CPUAndGPUComparison cpuAndGPUComparison = new CPUAndGPUComparison();
		cpuAndGPUComparison.compare(builder);
	}

}
