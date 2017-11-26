package com.github.neuralnetworks.analyzing.network.comparison;

import java.util.Random;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.input.RandomInputProvider;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * @author tmey
 */
public class EnvironmentConfigurationComparison
{
	private NetworkActivationAndWeightComparison similarNetworkWeightsComparison = new NetworkActivationAndWeightComparison();


	public void compareNetworkWithDiffEnvConf(NeuralNetworkBuilder builder, RuntimeConfiguration conf1, RuntimeConfiguration conf2) throws DifferentNetworksException
	{

		// reset builder to create same version again
		if (!builder.resetRandomSeed())
		{
			throw new IllegalStateException("It is not possible to reset the seed of the random initializer! Check your network definition if it uses the wrong weight initializer!");
		}

		// obtain values with conf1;
		Environment.getInstance().setRuntimeConfiguration(conf1);

		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair1 = builder.buildWithTrainer();
		BackPropagationTrainer<?> trainer1 = (BackPropagationTrainer<?>) neuralNetworkTrainerPair1.getRight();

		// extract initial weights
		ValuesProvider conf1Weights = ((NeuralNetworkImpl) trainer1.getNeuralNetwork()).getProperties().getParameter(Constants.WEIGHTS_PROVIDER);
		ValuesProvider originalWeights = TensorFactory.duplicate(conf1Weights);
		TensorFactory.copy(conf1Weights, originalWeights);

		// extract one train instance

		TrainingInputProvider oneInstanceProvider = getOneInstance(trainer1, trainer1.getTrainingInputProvider());
		trainer1.setTrainingInputProvider(oneInstanceProvider);

		trainer1.setEpochs(1);
		trainer1.train();


		// reset builder to create same version again
		if (!builder.resetRandomSeed())
		{
			throw new IllegalStateException("It is not possible to reset the seed of the random initializer! Check your network definition if it uses the wrong weight initializer!");
		}

		// obtain values with conf2;
		Environment.getInstance().setRuntimeConfiguration(conf2);

		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair2 = builder.buildWithTrainer();
		BackPropagationTrainer<?> trainer2 = (BackPropagationTrainer<?>) neuralNetworkTrainerPair2.getRight();
		trainer2.setTrainingInputProvider(oneInstanceProvider);

		trainer2.setEpochs(1);

		trainer2.train();

		similarNetworkWeightsComparison.compareTrainedNetworks(trainer1, trainer2, originalWeights);
	}


	private static TrainingInputProvider getOneInstance(Trainer<?> trainer, TrainingInputProvider inputProvider)
	{
		ValuesProvider results = TensorFactory.tensorProvider(trainer.getNeuralNetwork(), 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor targetTensor = TensorFactory.tensor(results.get(trainer.getNeuralNetwork().getOutputLayer()).getDimensions());
		Tensor inputTensor = TensorFactory.tensor(results.get(trainer.getNeuralNetwork().getInputLayer()).getDimensions());
		inputProvider.getNextInput(inputTensor);
		inputProvider.getNextTarget(targetTensor);

		float[] input = inputTensor.getElements();
		float[] output = targetTensor.getElements();

		return new SimpleInputProvider(new float[][] { input }, new float[][] { output });
	}

	public NetworkActivationAndWeightComparison getSimilarNetworkWeightsComparison()
	{
		return similarNetworkWeightsComparison;
	}

	public void setSimilarNetworkWeightsComparison(NetworkActivationAndWeightComparison similarNetworkWeightsComparison)
	{
		this.similarNetworkWeightsComparison = similarNetworkWeightsComparison;
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


		RuntimeConfiguration conf1 = new RuntimeConfiguration();
		conf1.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);

		// conf2 (for tests)
		RuntimeConfiguration conf2 = new RuntimeConfiguration();
		conf2.setCalculationProvider(RuntimeConfiguration.CalculationProvider.OPENCL);
		conf2.setUseDataSharedMemory(false);
		conf2.setUseWeightsSharedMemory(false);
		conf2.getOpenCLConfiguration().setAggregateOperations(true);
		conf2.getOpenCLConfiguration().setUseOptionsString(false);
		conf2.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
		conf2.getOpenCLConfiguration().setPushToDeviceBeforeOperation(false);
		conf2.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(false);
		conf2.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);

		EnvironmentConfigurationComparison environmentConfigurationComparison = new EnvironmentConfigurationComparison();
		environmentConfigurationComparison.compareNetworkWithDiffEnvConf(builder, conf1, conf2);
	}

}
