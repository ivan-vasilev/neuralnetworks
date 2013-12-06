package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DSoftReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSoftReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiStochasticBinary;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiTanh;

/**
 * Factory class for neural networks
 */
public class NNFactory {

    public static MultiLayerPerceptron mlp(Layer[] layers, boolean addBias) {
	if (layers.length <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}

	MultiLayerPerceptron result = new MultiLayerPerceptron();
	for (int i = 0; i < layers.length; i++) {
	    result.addLayer(layers[i], addBias);
	}

	return result;
    }

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
	result.addLayer(new Layer(layers[0], new AparapiSoftReLU()), false);
	for (int i = 1; i < layers.length; i++) {
	    result.addLayer(new Layer(layers[i], new AparapiSoftReLU()), addBias);
	}
	
	return result;
    }

    public static MultiLayerPerceptron mlpTanh(int[] layers, boolean addBias) {
	if (layers.length <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}

	MultiLayerPerceptron result = new MultiLayerPerceptron();
	result.addLayer(new Layer(layers[0], new AparapiTanh()), false);
	for (int i = 1; i < layers.length; i++) {
	    result.addLayer(new Layer(layers[i], new AparapiTanh()), addBias);
	}

	return result;
    }

    public static MultiLayerPerceptron autoencoderSigmoid(int visibleCount, int hiddenCount, boolean addBias) {
	return mlpSigmoid(new int[] {visibleCount,  hiddenCount, visibleCount}, addBias);
    }
    
    public static MultiLayerPerceptron autoencoderReLU(int visibleCount, int hiddenCount, boolean addBias) {
	return mlpRelu(new int[] {visibleCount,  hiddenCount, visibleCount}, addBias);
    }

    public static MultiLayerPerceptron autoencoderTanh(int visibleCount, int hiddenCount, boolean addBias) {
	return mlpTanh(new int[] {visibleCount,  hiddenCount, visibleCount}, addBias);
    }

    public static RBM rbmSigmoidSigmoid(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount, new AparapiSigmoid()), new Layer(hiddenCount, new AparapiSigmoid()), addBias, addBias);
    }

    public static RBM rbmReluRelu(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount, new AparapiSoftReLU()), new Layer(hiddenCount, new AparapiSoftReLU()), addBias, addBias);
    }

    public static RBM rbmTanhTanh(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount, new AparapiTanh()), new Layer(hiddenCount, new AparapiTanh()), addBias, addBias);
    }

    public static RBM rbmSigmoidBinary(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount, new AparapiSigmoid()), new Layer(hiddenCount, new AparapiStochasticBinary()), addBias, addBias);
    }

    public static RBM rbmReluBinary(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount, new AparapiSoftReLU()), new Layer(hiddenCount, new AparapiStochasticBinary()), addBias, addBias);
    }

    public static RBM rbmTanhBinary(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount, new AparapiTanh()), new Layer(hiddenCount, new AparapiStochasticBinary()), addBias, addBias);
    }

    public static SupervisedRBM srbmSigmoidSigmoid(int visibleCount, int hiddenCount, int dataCount, boolean addBias) {
	return new SupervisedRBM(new Layer(visibleCount, new AparapiSigmoid()), new Layer(hiddenCount, new AparapiSigmoid()), new Layer(dataCount), addBias, addBias);
    }

    public static SupervisedRBM srbmSigmoidBinary(int visibleCount, int hiddenCount, int dataCount, boolean addBias) {
	return new SupervisedRBM(new Layer(visibleCount, new AparapiSigmoid()), new Layer(hiddenCount, new AparapiStochasticBinary()), new Layer(dataCount), addBias, addBias);
    }

    public static SupervisedRBM srbmReluBinary(int visibleCount, int hiddenCount, int dataCount, boolean addBias) {
	return new SupervisedRBM(new Layer(visibleCount, new AparapiSoftReLU()), new Layer(hiddenCount, new AparapiStochasticBinary()), new Layer(dataCount), addBias, addBias);
    }

    public static SupervisedRBM srbmReluRelu(int visibleCount, int hiddenCount, int dataCount, boolean addBias) {
	return new SupervisedRBM(new Layer(visibleCount, new AparapiSoftReLU()), new Layer(hiddenCount, new AparapiSoftReLU()), new Layer(dataCount), addBias, addBias);
    }

    public static SupervisedRBM srbmTanhTanh(int visibleCount, int hiddenCount, int dataCount, boolean addBias) {
	return new SupervisedRBM(new Layer(visibleCount, new AparapiTanh()), new Layer(hiddenCount, new AparapiTanh()), new Layer(dataCount), addBias, addBias);
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
	    result.addLayer(new Layer(layers[i], new AparapiSoftReLU()), addBias);
	}
	
	return result;
    }

    public static DBN dbnTanh(int[] layers, boolean addBias) {
	if (layers.length <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}
	
	DBN result = new DBN();
	for (int i = 0; i < layers.length; i++) {
	    result.addLayer(new Layer(layers[i], new AparapiTanh()), addBias);
	}
	
	return result;
    }
    
    public static StackedAutoencoder saeSigmoid(int levels, int visibleCount, int hiddenCount, boolean addBias) {
	if (levels <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}

	StackedAutoencoder result = new StackedAutoencoder(new Layer(visibleCount, new AparapiSigmoid()));
	for (int i = 0; i < levels; i++) {
	    result.addLevel(new Layer(hiddenCount, new AparapiSigmoid()), new Layer(visibleCount, new AparapiSigmoid()), addBias);
	}

	return result;
    }

    public static StackedAutoencoder saeReLU(int levels, int visibleCount, int hiddenCount, boolean addBias) {
	if (levels <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}
	
	StackedAutoencoder result = new StackedAutoencoder(new Layer(visibleCount, new AparapiSoftReLU()));
	for (int i = 0; i < levels; i++) {
	    result.addLevel(new Layer(hiddenCount, new AparapiSoftReLU()), new Layer(visibleCount, new AparapiSoftReLU()), addBias);
	}

	return result;
    }

    public static StackedAutoencoder saeTanh(int levels, int visibleCount, int hiddenCount, boolean addBias) {
	if (levels <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}
	
	StackedAutoencoder result = new StackedAutoencoder(new Layer(visibleCount, new AparapiTanh()));
	for (int i = 0; i < levels; i++) {
	    result.addLevel(new Layer(hiddenCount, new AparapiTanh()), new Layer(visibleCount, new AparapiTanh()), addBias);
	}
	
	return result;
    }

    public static Conv2DConnection convSigmoidConnection(ConvGridLayer inputLayer, int featureMapRows, int featureMapColumns, int featureMaps) {
	Conv2DConnection result = new Conv2DConnection(inputLayer, new AparapiConv2DSigmoid());

	for (int i = 0; i < featureMaps; i++) {
	    result.addFilter(new Matrix(featureMapRows, featureMapColumns));
	}

	return result;
    }
    
    public static Conv2DConnection convSoftReLUConnection(ConvGridLayer inputLayer, int featureMapRows, int featureMapColumns, int featureMaps) {
	Conv2DConnection result = new Conv2DConnection(inputLayer, new AparapiConv2DSoftReLU());

	for (int i = 0; i < featureMaps; i++) {
	    result.addFilter(new Matrix(featureMapRows, featureMapColumns));
	}

	return result;
    }
}
