package com.github.neuralnetworks.usage;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.OneInputProvider;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputDataImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.TestingFinishedEvent;
import com.github.neuralnetworks.training.events.TestingStartedEvent;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.UniqueList;

/**
 * This class allow to use a neural network to classify a given input vector and receive the output vector.
 * In a future version this class will also use batches of different size to classify several input vectors at the
 * same time with high speed.
 *
 * @author tmey
 */
public class NeuralNetworkUsage
{

	private final NeuralNetworkImpl neuralNetwork;
	private final BackPropagationTrainer<?> trainer;
	private int batchSize;

	public NeuralNetworkUsage(NeuralNetworkImpl neuralNetwork)
	{
		this(neuralNetwork, 128);
	}

	public NeuralNetworkUsage(NeuralNetworkImpl neuralNetwork, int batchSize)
	{

		this.neuralNetwork = neuralNetwork;
		this.batchSize = batchSize;

		trainer = TrainerFactory.backPropagation(neuralNetwork, null, null, new MultipleNeuronsOutputError(), null
				, 0.01f, 0.5f, 0.0f, 0f, 0.0f, batchSize, batchSize, 1);
		trainer.triggerEvent(new TestingStartedEvent(trainer));

	}

	/**
	 * currently don't use larger batch size!
	 *
	 * @param listOfInputs
	 * @return
	 */
	public List<float[]> classifyListOfVectors(List<float[]> listOfInputs)
	{
		List<float[]> listOfTargets = new ArrayList<>();

		for (float[] input : listOfInputs)
		{
			listOfTargets.add(classifyVector(input));
		}

		return listOfTargets;
	}

	/**
	 * create a Tensor with the input vector, classify it, trim the result and returns the output vector of the network.
	 *
	 * @param input
	 * @return
	 */
	public float[] classifyVector(float[] input)
	{
		return classifyVectorWithActivationOutput(input).getLeft();
	}

	public Pair<float[], ValuesProvider> classifyVectorWithActivationOutput(float[] input)
	{

		// prepare input
		ValuesProvider results = TensorFactory.tensorProvider(neuralNetwork, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor targetTensor = TensorFactory.tensor(results.get(neuralNetwork.getOutputLayer()).getDimensions());

		OneInputProvider oneInputProvider = new OneInputProvider(input, new float[results.get(neuralNetwork.getOutputLayer()).getDimensions()[0]]);

		TrainingInputData inputTensor = new TrainingInputDataImpl(results.get(neuralNetwork.getInputLayer()), targetTensor);

		oneInputProvider.populateNext(inputTensor);

		// prepare network and train
		Set<Layer> calculatedLayers = new UniqueList<>();
		calculatedLayers.add(neuralNetwork.getInputLayer());
		neuralNetwork.getLayerCalculator().calculate(neuralNetwork, neuralNetwork.getOutputLayer(), calculatedLayers, results);

		Matrix networkOutput = results.get(neuralNetwork.getOutputLayer());

		// trim output
		float[] target = new float[networkOutput.getSize()];
		System.arraycopy(networkOutput.getElements(), networkOutput.getStartIndex(), target, 0, networkOutput.getSize());

		return new Pair<>(target, results);
	}

	public int getBatchSize()
	{
		return batchSize;
	}

	public void setBatchSize(int batchSize)
	{
		if (batchSize >= 0)
		{
			throw new IllegalArgumentException("The batch sze must be greater than 0!");
		}
		this.batchSize = batchSize;
	}

	@Override
	protected void finalize() throws Throwable
	{
		super.finalize();

		trainer.triggerEvent(new TestingFinishedEvent(trainer));
	}
}
