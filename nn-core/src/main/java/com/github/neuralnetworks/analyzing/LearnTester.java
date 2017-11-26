package com.github.neuralnetworks.analyzing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.input.OneInputProvider;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.usage.NeuralNetworkUsage;
import com.github.neuralnetworks.util.Environment;

/**
 * @author tmey
 */
public class LearnTester
{

	private static final Logger logger = LoggerFactory.getLogger(LearnTester.class);

	public boolean testIfNetworkLearns(Trainer<?> trainer, int iterations, TrainingInputProvider inputProvider)
	{

		ValuesProvider results = TensorFactory.tensorProvider(trainer.getNeuralNetwork(), 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor targetTensor = TensorFactory.tensor(results.get(trainer.getNeuralNetwork().getOutputLayer()).getDimensions());
		Tensor inputTensor = TensorFactory.tensor(results.get(trainer.getNeuralNetwork().getInputLayer()).getDimensions());
		inputProvider.getNextInput(inputTensor);
		inputProvider.getNextTarget(targetTensor);

		float[] input = inputTensor.getElements();
		float[] output = targetTensor.getElements();

		return testIfNetworkLearns(trainer, iterations, input, output);
	}

	public boolean testIfNetworkLearns(Trainer<?> trainer, int iterations, float[] input, float[] output)
	{


		OneInputProvider oneInputProvider = new OneInputProvider(input, output);

		// classify instance
		NeuralNetworkUsage neuralNetworkUsage = new NeuralNetworkUsage((NeuralNetworkImpl) trainer.getNeuralNetwork());
		float[] firstPrediction = neuralNetworkUsage.classifyVector(input);

		// train instance
		trainer.setTrainingInputProvider(oneInputProvider);
		trainer.setTestingInputProvider(null);
		trainer.setEpochs(iterations);

		trainer.train();

		// classify instance
		float[] secondPrediction = neuralNetworkUsage.classifyVector(input);

		// compare results
		double firstError = getEuclideanDistance(output, firstPrediction);
		double secondError = getEuclideanDistance(output, secondPrediction);

		logger.info("first eucl. error: " + firstError + " > second eucl. error " + secondError);

		String connectionWeights = ConnectionAnalysis.analyseConnectionWeights((NeuralNetworkImpl) trainer.getNeuralNetwork());
		logger.info("weights:\n" + connectionWeights);
		if (connectionWeights.contains("NaN"))
		{
			throw new IllegalStateException("infinite number problem!!!");
		}

		if (firstError > secondError)
		{
			return true;
		} else
		{
			return false;
		}
	}

	private double getEuclideanDistance(float[] target, float[] prediction)
	{
		if (target.length != prediction.length)
		{
			throw new IllegalArgumentException("target and prediction must have the same size (t: " + target.length + ", p: " + prediction.length + ")");
		}

		float value = 0;

		for (int i = 0; i < target.length; i++)
		{
			value += Math.pow(target[i] - prediction[i], 2);
		}

		return Math.sqrt(value);
	}

}
