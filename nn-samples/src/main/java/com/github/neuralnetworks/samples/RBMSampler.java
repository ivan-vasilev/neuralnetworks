package com.github.neuralnetworks.samples;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;
import com.github.neuralnetworks.input.MeanInputModifier;
import com.github.neuralnetworks.input.ScalingInputModifier;
import com.github.neuralnetworks.input.mnist.MnistInputConverter;
import com.github.neuralnetworks.input.mnist.MnistInputProvider;
import com.github.neuralnetworks.input.mnist.MnistTargetMultiNeuronOutputConverter;
import com.github.neuralnetworks.outputerror.MnistMultipleNeuronsOutputError;
import com.github.neuralnetworks.testing.Sampler;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.rbm.CDAparapiTrainer;
import com.github.neuralnetworks.training.rbm.CDAparapiTrainerBase;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

public class RBMSampler extends Sampler {

    public RBMSampler() {
	super();

	MnistInputConverter inputTrainingConverter = new MnistInputConverter();
	inputTrainingConverter.addModifier(new MeanInputModifier());
	inputTrainingConverter.addModifier(new ScalingInputModifier(255));


	MnistInputConverter inputTestingConverter = new MnistInputConverter();
	inputTestingConverter.addModifier(new MeanInputModifier());
	inputTestingConverter.addModifier(new ScalingInputModifier(255));

	MnistTargetMultiNeuronOutputConverter targetConverter = new MnistTargetMultiNeuronOutputConverter();
	MnistInputProvider training = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 10, inputTrainingConverter, targetConverter);
	MnistInputProvider testing = new MnistInputProvider("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 1, inputTestingConverter, targetConverter);

	ConnectionCalculator cc = new AparapiSigmoid();
	RBM rbm = new RBM(new Layer(training.getRows() * training.getCols(), cc), new Layer(10, cc), false, false);

	Properties trainerProperties = new Properties();
	trainerProperties.setParameter(Constants.NEURAL_NETWORK, rbm);
	trainerProperties.setParameter(Constants.TRAINING_INPUT_PROVIDER, training);
	trainerProperties.setParameter(Constants.TESTING_INPUT_PROVIDER, testing);
	trainerProperties.setParameter(Constants.LAYER_CALCULATOR, new LayerCalculatorImpl());
	trainerProperties.setParameter(Constants.LEARNING_RATE, 0.01f);
	trainerProperties.setParameter(Constants.OUTPUT_ERROR, new MnistMultipleNeuronsOutputError());
	trainerProperties.setParameter(Constants.RANDOM_INITIALIZER, new MersenneTwisterRandomInitializer(-0.1f, 0.2f));
	CDAparapiTrainerBase trainer = new CDAparapiTrainer(trainerProperties);
	trainingConfigurations.add(trainer);
    }
}
