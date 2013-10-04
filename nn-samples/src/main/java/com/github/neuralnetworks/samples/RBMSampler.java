package com.github.neuralnetworks.samples;

import com.github.neuralnetworks.activation.AparapiSigmoidFunction;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.input.MnistInputProvider;
import com.github.neuralnetworks.neuroninput.AparapiWeightedSum;
import com.github.neuralnetworks.testing.Sampler;
import com.github.neuralnetworks.training.ContrastiveDivergenceAparapiTrainer;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

public class RBMSampler extends Sampler {

	public RBMSampler() {
		super();

		MnistInputProvider training = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
		MnistInputProvider testing = new MnistInputProvider("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

		Properties rbmProperties = new Properties();
		rbmProperties.setParameter(Constants.HIDDEN_COUNT, 10);
		rbmProperties.setParameter(Constants.VISIBLE_COUNT, training.getRows() * training.getCols());
		rbmProperties.setParameter(Constants.INPUT_FUNCTION, new AparapiWeightedSum());
		rbmProperties.setParameter(Constants.ACTIVATION_FUNCTION, new AparapiSigmoidFunction());
		rbmProperties.setParameter(Constants.ADD_BIAS, true);
		RBM rbm = new RBM(rbmProperties);

		ContrastiveDivergenceAparapiTrainer trainer = new ContrastiveDivergenceAparapiTrainer();
	}
}
