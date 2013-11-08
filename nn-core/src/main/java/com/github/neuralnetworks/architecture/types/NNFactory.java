package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiStochasticBinary;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;

/**
 * Factory class for neural networks
 */
public class NNFactory {

    public static MultiLayerPerceptron sigmoidMLP(int[] layers, boolean addBias) {
	if (layers.length <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}

	MultiLayerPerceptron result = new MultiLayerPerceptron();
	result.addLayer(new Layer(layers[0], new AparapiSigmoid()), false);
	for (int i = 1; i < layers.length; i++) {
	    result.addLayer(new Layer(layers[i], new AparapiSigmoid()), addBias);
	}

	return result;
    }

    public static MultiLayerPerceptron reluMLP(int[] layers, boolean addBias) {
	if (layers.length <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}
	
	MultiLayerPerceptron result = new MultiLayerPerceptron();
	result.addLayer(new Layer(layers[0], new AparapiReLU()), false);
	for (int i = 1; i < layers.length; i++) {
	    result.addLayer(new Layer(layers[i], new AparapiReLU()), addBias);
	}
	
	return result;
    }

    public static RBM sigmoidBinaryRBM(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount, new AparapiSigmoid()), new Layer(hiddenCount, new AparapiStochasticBinary()), addBias, addBias);
    }

    public static RBM reluBinaryRBM(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount, new AparapiReLU()), new Layer(hiddenCount, new AparapiStochasticBinary()), addBias, addBias);
    }

    public static SupervisedRBM sigmoidBinarySRBM(int visibleCount, int hiddenCount, int dataCount, boolean addBias) {
	return new SupervisedRBM(new Layer(visibleCount, new AparapiSigmoid()), new Layer(hiddenCount, new AparapiStochasticBinary()), new Layer(dataCount), addBias, addBias);
    }

    public static SupervisedRBM reluBinarySRBM(int visibleCount, int hiddenCount, int dataCount, boolean addBias) {
	return new SupervisedRBM(new Layer(visibleCount, new AparapiReLU()), new Layer(hiddenCount, new AparapiStochasticBinary()), new Layer(dataCount), addBias, addBias);
    }
}
