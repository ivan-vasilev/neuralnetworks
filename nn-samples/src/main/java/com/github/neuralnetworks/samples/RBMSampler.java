package com.github.neuralnetworks.samples;

import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.input.MnistInputProvider;
import com.github.neuralnetworks.neuronfunctions.AparapiSigmoidByRows;
import com.github.neuralnetworks.neuronfunctions.AparapiSigmoidByRows.AparapiSigmoidByColumns;
import com.github.neuralnetworks.neuronfunctions.RepeaterFunction;
import com.github.neuralnetworks.outputerror.MnistOutputError;
import com.github.neuralnetworks.testing.Sampler;
import com.github.neuralnetworks.training.ContrastiveDivergenceAparapiTrainer;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

public class RBMSampler extends Sampler {

	public RBMSampler() {
		super();

		MnistInputProvider training = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 10);
		MnistInputProvider testing = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1);

		Properties rbmProperties = new Properties();
		rbmProperties.setParameter(Constants.HIDDEN_COUNT, 10);
		rbmProperties.setParameter(Constants.VISIBLE_COUNT, training.getRows() * training.getCols());
		rbmProperties.setParameter(Constants.FORWARD_INPUT_FUNCTION, new AparapiSigmoidByRows());
		rbmProperties.setParameter(Constants.BACKWARD_INPUT_FUNCTION, new AparapiSigmoidByColumns());
		rbmProperties.setParameter(Constants.ACTIVATION_FUNCTION, new RepeaterFunction());
		rbmProperties.setParameter(Constants.ADD_BIAS, true);
		RBM rbm = new RBM(rbmProperties);

		Properties trainerProperties = new Properties();
		trainerProperties.setParameter(Constants.NEURAL_NETWORK, rbm);
		trainerProperties.setParameter(Constants.TRAINING_INPUT_PROVIDER, training);
		trainerProperties.setParameter(Constants.TESTING_INPUT_PROVIDER, testing);
		trainerProperties.setParameter(Constants.LAYER_CALCULATOR, new LayerCalculatorImpl());
		trainerProperties.setParameter(Constants.MINI_BATCH_SIZE, 10);
		trainerProperties.setParameter(Constants.LEARNING_RATE, 0.01f);
		trainerProperties.setParameter(Constants.OUTPUT_ERROR, new MnistOutputError());
		ContrastiveDivergenceAparapiTrainer trainer = new ContrastiveDivergenceAparapiTrainer(trainerProperties);
		trainingConfigurations.add(trainer);
	}
}
