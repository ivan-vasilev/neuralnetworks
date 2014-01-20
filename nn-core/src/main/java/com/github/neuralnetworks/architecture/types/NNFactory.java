package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSoftReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiStochasticBinary;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiTanh;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;
import com.github.neuralnetworks.training.random.AparapiXORShiftInitializer;

/**
 * Factory class for neural networks
 */
public class NNFactory {

    public static MultiLayerPerceptron mlp(int[] layers, boolean addBias) {
	if (layers.length <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}

	MultiLayerPerceptron result = new MultiLayerPerceptron();
	for (int i = 0; i < layers.length; i++) {
	    result.addLayer(new Layer(layers[i]), addBias);
	}

	return result;
    }

    public static void nnSigmoid(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	nn.setLayerCalculator(lc);
	for (Layer l : nn.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != nn.getInputLayer()) {
		    if (outputCC != null && nn.getOutputLayer() == l) {
			lc.addConnectionCalculator(l, outputCC);
		    } else {
			lc.addConnectionCalculator(l, new AparapiSigmoid());
		    }
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}
    }

    public static void nnSoftRelu(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	nn.setLayerCalculator(lc);
	for (Layer l : nn.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != nn.getInputLayer()) {
		    if (outputCC != null && nn.getOutputLayer() == l) {
			lc.addConnectionCalculator(l, outputCC);
		    } else {
			lc.addConnectionCalculator(l, new AparapiSoftReLU());
		    }
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}
    }
    
    public static void nnRelu(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	nn.setLayerCalculator(lc);
	for (Layer l : nn.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != nn.getInputLayer()) {
		    if (outputCC != null && nn.getOutputLayer() == l) {
			lc.addConnectionCalculator(l, outputCC);
		    } else {
			lc.addConnectionCalculator(l, new AparapiReLU());
		    }
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}
    }
    
    public static void nnTanh(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	nn.setLayerCalculator(lc);
	for (Layer l : nn.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != nn.getInputLayer()) {
		    if (outputCC != null && nn.getOutputLayer() == l) {
			lc.addConnectionCalculator(l, outputCC);
		    } else {
			lc.addConnectionCalculator(l, new AparapiTanh());
		    }
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}
    }

    public static MultiLayerPerceptron mlpSigmoid(int[] layers, boolean addBias) {
	MultiLayerPerceptron result = mlp(layers, addBias);
	nnSigmoid(result, null);
	return result;
    }

    public static MultiLayerPerceptron mlpSoftRelu(int[] layers, boolean addBias, ConnectionCalculator outputCC) {
	MultiLayerPerceptron result = mlp(layers, addBias);
	nnSoftRelu(result, outputCC);
	return result;
    }

    public static MultiLayerPerceptron mlpRelu(int[] layers, boolean addBias, ConnectionCalculator outputCC) {
	MultiLayerPerceptron result = mlp(layers, addBias);
	nnRelu(result, outputCC);
	return result;
    }

    public static MultiLayerPerceptron mlpTanh(int[] layers, boolean addBias, ConnectionCalculator outputCC) {
	MultiLayerPerceptron result = mlp(layers, addBias);
	nnTanh(result, outputCC);
	return result;
    }

    public static Autoencoder autoencoder(int visibleCount, int hiddenCount, boolean addBias) {
	return new Autoencoder(new Layer(visibleCount), new Layer(hiddenCount), new Layer(visibleCount), addBias);
    }

    public static Autoencoder autoencoderSigmoid(int visibleCount, int hiddenCount, boolean addBias) {
	Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
	nnSigmoid(ae, null);
	return ae;
    }

    public static Autoencoder autoencoderSoftReLU(int visibleCount, int hiddenCount, boolean addBias, ConnectionCalculator outputCC) {
	Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
	nnSoftRelu(ae, outputCC);
	return ae;
    }
    
    public static Autoencoder autoencoderReLU(int visibleCount, int hiddenCount, boolean addBias, ConnectionCalculator outputCC) {
	Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
	nnRelu(ae, outputCC);
	return ae;
    }

    public static Autoencoder autoencoderTanh(int visibleCount, int hiddenCount, boolean addBias, ConnectionCalculator outputCC) {
	Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
	nnTanh(ae, outputCC);
	return ae;
    }

    public static RBM rbm(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount), new Layer(hiddenCount), addBias, addBias);
    }

    public static void rbmSigmoidSigmoid(RBM rbm) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiSigmoid());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiSigmoid());
	populateBiasLayers(lc, rbm);
    }

    public static void rbmSoftReluSoftRelu(RBM rbm) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiSoftReLU());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiSoftReLU());
	populateBiasLayers(lc, rbm);
    }
    
    public static void rbmReluRelu(RBM rbm) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiReLU());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiReLU());
	populateBiasLayers(lc, rbm);
    }

    public static void rbmTanhTanh(RBM rbm) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiTanh());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiTanh());
	populateBiasLayers(lc, rbm);
    }

    public static void rbmSigmoidBinary(RBM rbm) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiSigmoid());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiStochasticBinary(new AparapiXORShiftInitializer()));
	populateBiasLayers(lc, rbm);
    }

    public static void rbmSoftReluBinary(RBM rbm) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiSoftReLU());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiStochasticBinary(new AparapiXORShiftInitializer()));
	populateBiasLayers(lc, rbm);
    }
    
    public static void rbmReluBinary(RBM rbm) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiReLU());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiStochasticBinary(new AparapiXORShiftInitializer()));
	populateBiasLayers(lc, rbm);
    }

    public static void rbmTanhBinary(RBM rbm) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiTanh());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiStochasticBinary(new AparapiXORShiftInitializer()));
	populateBiasLayers(lc, rbm);
    }

    public static DBN dbn(int[] layers, boolean addBias) {
	if (layers.length <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}

	DBN result = new DBN();
	for (int i = 0; i < layers.length; i++) {
	    result.addLevel(new Layer(layers[i]), addBias);
	}

	return result;
    }

    public static DBN dbnSigmoid(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);
	nnSigmoid(result, null);
	return result;
    }
    
    public static DBN dbnSoftReLU(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);
	nnSoftRelu(result, null);
	return result;
    }
    
    public static DBN dbnReLU(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);
	nnRelu(result, null);
	return result;
    }

    public static DBN dbnTanh(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);
	nnTanh(result, null);
	return result;
    }
    public static StackedAutoencoder sae(int[] layers, boolean addBias) {
	if (layers == null || layers.length <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}

	StackedAutoencoder result = new StackedAutoencoder(new Layer(layers[0]));
	for (int i = 1; i < layers.length; i++) {
	    result.addLevel(new Layer(layers[i]), addBias);
	}

	return result;
    }

    public static StackedAutoencoder saeSigmoid(int[] layers, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);
	nnSigmoid(sae, null);
	return sae;
    }

    public static StackedAutoencoder saeSoftReLU(int[] layers, int hiddenCount, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);
	nnSoftRelu(sae, null);
	return sae;
    }

    public static StackedAutoencoder saeReLU(int[] layers, int hiddenCount, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);
	nnRelu(sae, null);
	return sae;
    }

    public static StackedAutoencoder saeTanh(int[] layers, int hiddenCount, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);
	nnTanh(sae, null);
	return sae;
    }

    public static void populateBiasLayers(LayerCalculatorImpl lc, NeuralNetwork nn) {
	for (Layer l : nn.getLayers()) {
	    lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	}
    }
}
