package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DSoftReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DTanh;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSoftReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiTanh;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSumConnectionCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorConv;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorFullyConnected;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.SoftmaxFunction;
import com.github.neuralnetworks.util.Util;

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
    
    public static void nnWeightedSum(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	nn.setLayerCalculator(lc);
	for (Layer l : nn.getLayers()) {
	    if (!Util.isBias(l)) {
		if (l != nn.getInputLayer()) {
		    if (outputCC != null && nn.getOutputLayer() == l) {
			lc.addConnectionCalculator(l, outputCC);
		    } else if (l instanceof ConvGridLayer) {
			lc.addConnectionCalculator(l, new ConnectionCalculatorConv());
		    } else {
			lc.addConnectionCalculator(l, new ConnectionCalculatorFullyConnected());
		    }
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}
    }

    public static LayerCalculatorImpl nnSigmoid(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	for (Layer l : nn.getLayers()) {
	    if (!Util.isBias(l)) {
		if (l != nn.getInputLayer()) {
		    if (outputCC != null && nn.getOutputLayer() == l) {
			lc.addConnectionCalculator(l, outputCC);
		    } else if (l instanceof ConvGridLayer) {
			lc.addConnectionCalculator(l, new AparapiConv2DSigmoid());
		    } else {
			lc.addConnectionCalculator(l, new AparapiSigmoid());
		    }
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return lc;
    }

    public static LayerCalculatorImpl nnSoftRelu(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	for (Layer l : nn.getLayers()) {
	    if (!Util.isBias(l)) {
		if (l != nn.getInputLayer()) {
		    if (nn.getOutputLayer() == l) {
			if (outputCC != null) {
			    lc.addConnectionCalculator(l, outputCC);
			} else {
			    AparapiSoftReLU c = new AparapiSoftReLU();
			    c.addActivationFunction(new SoftmaxFunction());
			    lc.addConnectionCalculator(l, c);
			}
		    } else if (l instanceof ConvGridLayer) {
			lc.addConnectionCalculator(l, new AparapiConv2DSoftReLU());
		    } else {
			lc.addConnectionCalculator(l, new AparapiSoftReLU());
		    }
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return lc;
    }
    
    public static LayerCalculatorImpl nnRelu(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	for (Layer l : nn.getLayers()) {
	    if (!Util.isBias(l)) {
		if (l != nn.getInputLayer()) {
		    if (nn.getOutputLayer() == l) {
			if (outputCC != null) {
			    lc.addConnectionCalculator(l, outputCC);
			} else {
			    AparapiReLU c = new AparapiReLU();
			    c.addActivationFunction(new SoftmaxFunction());
			    lc.addConnectionCalculator(l, c);
			}
		    } else if (l instanceof ConvGridLayer) {
			lc.addConnectionCalculator(l, new AparapiConv2DReLU());
		    } else {
			lc.addConnectionCalculator(l, new AparapiReLU());
		    }
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return lc;
    }
    
    public static LayerCalculatorImpl nnTanh(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	for (Layer l : nn.getLayers()) {
	    if (!Util.isBias(l)) {
		if (l != nn.getInputLayer()) {
		    if (outputCC != null && nn.getOutputLayer() == l) {
			lc.addConnectionCalculator(l, outputCC);
		    } else if (l instanceof ConvGridLayer) {
			lc.addConnectionCalculator(l, new AparapiConv2DTanh());
		    } else {
			lc.addConnectionCalculator(l, new AparapiTanh());
		    }
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return lc;
    }

    public static MultiLayerPerceptron mlpSigmoid(int[] layers, boolean addBias) {
	MultiLayerPerceptron result = mlp(layers, addBias);
	result.setLayerCalculator(nnSigmoid(result, null));
	return result;
    }

    public static MultiLayerPerceptron mlpSoftRelu(int[] layers, boolean addBias, ConnectionCalculator outputCC) {
	MultiLayerPerceptron result = mlp(layers, addBias);
	result.setLayerCalculator(nnSoftRelu(result, outputCC));
	return result;
    }

    public static MultiLayerPerceptron mlpRelu(int[] layers, boolean addBias, ConnectionCalculator outputCC) {
	MultiLayerPerceptron result = mlp(layers, addBias);
	result.setLayerCalculator(nnRelu(result, outputCC));
	return result;
    }

    public static MultiLayerPerceptron mlpTanh(int[] layers, boolean addBias, ConnectionCalculator outputCC) {
	MultiLayerPerceptron result = mlp(layers, addBias);
	result.setLayerCalculator(nnTanh(result, outputCC));
	return result;
    }

    public static Autoencoder autoencoder(int visibleCount, int hiddenCount, boolean addBias) {
	return new Autoencoder(new Layer(visibleCount), new Layer(hiddenCount), new Layer(visibleCount), addBias);
    }

    public static Autoencoder autoencoderSigmoid(int visibleCount, int hiddenCount, boolean addBias) {
	Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
	ae.setLayerCalculator(nnSigmoid(ae, null));
	return ae;
    }

    public static Autoencoder autoencoderSoftReLU(int visibleCount, int hiddenCount, boolean addBias, ConnectionCalculator outputCC) {
	Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
	ae.setLayerCalculator(nnSoftRelu(ae, outputCC));
	return ae;
    }
    
    public static Autoencoder autoencoderReLU(int visibleCount, int hiddenCount, boolean addBias, ConnectionCalculator outputCC) {
	Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
	ae.setLayerCalculator(nnRelu(ae, outputCC));
	return ae;
    }

    public static Autoencoder autoencoderTanh(int visibleCount, int hiddenCount, boolean addBias, ConnectionCalculator outputCC) {
	Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
	ae.setLayerCalculator(nnTanh(ae, outputCC));
	return ae;
    }

    public static RBM rbm(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(new Layer(visibleCount), new Layer(hiddenCount), addBias, addBias);
    }
    
    public static RBMLayerCalculator rbmWeightedSumWeightedSum(RBM rbm) {
	RBMLayerCalculator lc = new RBMLayerCalculator();
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiWeightedSumConnectionCalculator());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiWeightedSumConnectionCalculator());
	populateBiasLayers(lc, rbm);
	return lc;
    }

    public static RBMLayerCalculator rbmSigmoidSigmoid(RBM rbm) {
	RBMLayerCalculator lc = new RBMLayerCalculator();
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiSigmoid());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiSigmoid());
	populateBiasLayers(lc, rbm);
	return lc;
    }

    public static RBMLayerCalculator rbmSoftReluSoftRelu(RBM rbm) {
	RBMLayerCalculator lc = new RBMLayerCalculator();

	AparapiSoftReLU c1 = new AparapiSoftReLU();
	c1.addActivationFunction(new SoftmaxFunction());
	lc.addConnectionCalculator(rbm.getVisibleLayer(), c1);


	AparapiSoftReLU c2 = new AparapiSoftReLU();
	c2.addActivationFunction(new SoftmaxFunction());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), c2);

	populateBiasLayers(lc, rbm);

	return lc;
    }
    
    public static RBMLayerCalculator rbmReluRelu(RBM rbm) {
	RBMLayerCalculator lc = new RBMLayerCalculator();

	AparapiReLU c1 = new AparapiReLU();
	c1.addActivationFunction(new SoftmaxFunction());
	lc.addConnectionCalculator(rbm.getVisibleLayer(), c1);

	AparapiReLU c2 = new AparapiReLU();
	c2.addActivationFunction(new SoftmaxFunction());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), c2);

	populateBiasLayers(lc, rbm);

	return lc;
    }

    public static RBMLayerCalculator rbmTanhTanh(RBM rbm) {
	RBMLayerCalculator lc = new RBMLayerCalculator();
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiTanh());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiTanh());
	populateBiasLayers(lc, rbm);

	return lc;
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
	result.setLayerCalculator(nnSigmoid(result, null));
	return result;
    }
    
    public static DBN dbnSoftReLU(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);
	result.setLayerCalculator(nnSoftRelu(result, null));
	return result;
    }
    
    public static DBN dbnReLU(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);
	result.setLayerCalculator(nnRelu(result, null));
	return result;
    }

    public static DBN dbnTanh(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);
	result.setLayerCalculator(nnTanh(result, null));
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
	sae.setLayerCalculator(nnSigmoid(sae, null));
	return sae;
    }

    public static StackedAutoencoder saeSoftReLU(int[] layers, int hiddenCount, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);
	sae.setLayerCalculator(nnSoftRelu(sae, null));
	return sae;
    }

    public static StackedAutoencoder saeReLU(int[] layers, int hiddenCount, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);
	sae.setLayerCalculator(nnRelu(sae, null));
	return sae;
    }

    public static StackedAutoencoder saeTanh(int[] layers, int hiddenCount, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);
	sae.setLayerCalculator(nnTanh(sae, null));
	return sae;
    }

    public static void populateBiasLayers(LayerCalculatorImpl lc, NeuralNetwork nn) {
	for (Layer l : nn.getLayers()) {
	    if (Util.isBias(l)) {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}
    }
}
