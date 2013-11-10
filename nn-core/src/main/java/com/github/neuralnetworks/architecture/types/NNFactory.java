package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiStochasticBinary;

/**
 * Factory class for neural networks
 */
public class NNFactory {

    public static MultiLayerPerceptron mlpSigmoid(int[] layers, boolean addBias) {
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

    public static MultiLayerPerceptron mlpRelu(int[] layers, boolean addBias) {
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


    public static MultiLayerPerceptron autoencoderSigmoid(int visibleCount, int hiddenCount, boolean addBias) {
	return mlpSigmoid(new int[] {visibleCount,  hiddenCount, visibleCount}, addBias);
    }
    
    public static MultiLayerPerceptron autoencoderReLU(int visibleCount, int hiddenCount, boolean addBias) {
	return mlpRelu(new int[] {visibleCount,  hiddenCount, visibleCount}, addBias);
    }

    public static RBM rbmSigmoidSigmoid(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount, new AparapiSigmoid()), new Layer(hiddenCount, new AparapiSigmoid()), addBias, addBias);
    }

    public static RBM rbmReluRelu(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount, new AparapiReLU()), new Layer(hiddenCount, new AparapiReLU()), addBias, addBias);
    }

    public static RBM rbmSigmoidBinary(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount, new AparapiSigmoid()), new Layer(hiddenCount, new AparapiStochasticBinary()), addBias, addBias);
    }

    public static RBM rbmReluBinary(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount, new AparapiReLU()), new Layer(hiddenCount, new AparapiStochasticBinary()), addBias, addBias);
    }

    public static SupervisedRBM srbmSigmoidSigmoid(int visibleCount, int hiddenCount, int dataCount, boolean addBias) {
	return new SupervisedRBM(new Layer(visibleCount, new AparapiSigmoid()), new Layer(hiddenCount, new AparapiSigmoid()), new Layer(dataCount), addBias, addBias);
    }

    public static SupervisedRBM srbmSigmoidBinary(int visibleCount, int hiddenCount, int dataCount, boolean addBias) {
	return new SupervisedRBM(new Layer(visibleCount, new AparapiSigmoid()), new Layer(hiddenCount, new AparapiStochasticBinary()), new Layer(dataCount), addBias, addBias);
    }

    public static SupervisedRBM srbmReluBinary(int visibleCount, int hiddenCount, int dataCount, boolean addBias) {
	return new SupervisedRBM(new Layer(visibleCount, new AparapiReLU()), new Layer(hiddenCount, new AparapiStochasticBinary()), new Layer(dataCount), addBias, addBias);
    }
    
    public static SupervisedRBM srbmReluRelu(int visibleCount, int hiddenCount, int dataCount, boolean addBias) {
	return new SupervisedRBM(new Layer(visibleCount, new AparapiReLU()), new Layer(hiddenCount, new AparapiReLU()), new Layer(dataCount), addBias, addBias);
    }

    public static DBN dbnSigmoid(int[] layers, boolean addBias) {
	if (layers.length <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}

	DBN result = new DBN();
	for (int i = 0; i < layers.length; i++) {
	    result.addLayer(new Layer(layers[i], new AparapiSigmoid()), addBias);
	}

	return result;
    }
    
    public static DBN dbnReLU(int[] layers, boolean addBias) {
	if (layers.length <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}
	
	DBN result = new DBN();
	for (int i = 0; i < layers.length; i++) {
	    result.addLayer(new Layer(layers[i], new AparapiReLU()), addBias);
	}
	
	return result;
    }
}
