package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiAveragePooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DSoftReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DTanh;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiMaxPooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSoftReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiStochasticPooling2D;
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

    /**
     * Create convolutional network
     * @param layers
     * The first layer must have 3 parameters - rows, columns and filter count (usually 1)
     * Convolutional connections must have 4 parameters - kernelRows, kernelColumns, filters and stride. The first layer must be convolutional.
     * Subsampling connections must have 2 parameters - subsamplingRegionRows, subsamplingRegionCols
     * Regular layers must have 1 parameter - neuron count
     * 
     * @param addBias
     * @return neural network
     */
    public static NeuralNetworkImpl convNN(int[][] layers, boolean addBias) {
	if (layers.length <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}

	if (layers[0].length != 3) {
	    throw new IllegalArgumentException("first layer must be convolutional");
	}

	NeuralNetworkImpl result = new NeuralNetworkImpl();

	Layer prev = null;
	int prevUnitCount = layers[0][0] * layers[0][1] * layers[0][2];
	result.addLayer(prev = new Layer());
	for (int i = 1; i < layers.length; i++) {
	    int[] l = layers[i];
	    Layer newLayer = null;
	    Layer biasLayer = null;
	    if (l.length == 1) {
		new FullyConnected(prev, newLayer = new Layer(), prevUnitCount, l[0]);
		if (addBias) {
		    new FullyConnected(biasLayer = new Layer(), newLayer, 1, l[0]);
		}

		prevUnitCount = l[0];
	    } else if (l.length == 4 || l.length == 2) {
		Integer inputFMRows = null;
		Integer inputFMCols = null;
		Integer filters = null;
		if (i == 1) {
		    inputFMRows = layers[0][0];
		    inputFMCols = layers[0][1];
		    filters = layers[0][2];
		} else {
		    for (Connections c : prev.getConnections()) {
			if (c.getOutputLayer() == prev) {
			    if (c instanceof Conv2DConnection) {
				Conv2DConnection cc = (Conv2DConnection) c;
				inputFMRows = cc.getOutputFeatureMapRows();
				inputFMCols = cc.getOutputFeatureMapColumns();
				filters = cc.getOutputFilters();
				break;
			    } else if (c instanceof Subsampling2DConnection) {
				Subsampling2DConnection sc = (Subsampling2DConnection) c;
				inputFMRows = sc.getOutputFeatureMapRows();
				inputFMCols = sc.getOutputFeatureMapColumns();
				filters = sc.getFilters();
				break;
			    }
			}
		    }
		}

		if (l.length == 4) {
		    Conv2DConnection c = new Conv2DConnection(prev, newLayer = new Layer(), inputFMRows, inputFMCols, filters, l[0], l[1], l[2], l[3]);
		    if (addBias) {
			new Conv2DConnection(biasLayer = new Layer(), newLayer, c.getOutputFeatureMapRows(), c.getOutputFeatureMapColumns(), 1, 1, 1, l[2], l[3]);
		    }
		} else if (l.length == 2) {
		    new Subsampling2DConnection(prev, newLayer = new Layer(), inputFMRows, inputFMCols, l[0], l[1], filters);
		}
	    }

	    result.addLayer(newLayer);
	    if (biasLayer != null) {
		result.addLayer(biasLayer);
	    }

	    prev = newLayer;
	}

	return result;
    }

    /**
     * Multi layer perceptron with fully connected layers
     * @param layers - neuron count for each layer
     * @param addBias
     * @return
     */
    public static NeuralNetworkImpl mlp(int[] layers, boolean addBias) {
	if (layers.length <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}

	NeuralNetworkImpl result = new NeuralNetworkImpl();
	addFullyConnectedLayer(result, new Layer(), layers[0], layers[0], addBias);
	for (int i = 1; i < layers.length; i++) {
	    addFullyConnectedLayer(result, new Layer(), layers[i - 1], layers[i], addBias);
	}

	return result;
    }

    /**
     * Add fully connected layer to the output layer of the network
     * @param nn
     * @param layer
     * @param addBias
     */
    public static FullyConnected addFullyConnectedLayer(NeuralNetworkImpl nn, Layer layer, int inputUnitCount, int outputUnitCount, boolean addBias) {
	FullyConnected result = null;
	if (nn.addLayer(layer) && nn.getOutputLayer() != layer) {
	    result = new FullyConnected(nn.getOutputLayer(), layer, inputUnitCount, outputUnitCount);
	}

	if (addBias && nn.getInputLayer() != layer) {
	    Layer biasLayer = new Layer();
	    nn.addLayer(biasLayer);
	    return new FullyConnected(biasLayer, layer, 1, outputUnitCount);
	}

	return result;
    }

    public static LayerCalculatorImpl lcWeightedSum(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	for (Layer l : nn.getLayers()) {
	    if (!Util.isBias(l)) {
		if (outputCC != null && nn.getOutputLayer() == l) {
		    lc.addConnectionCalculator(l, outputCC);
		} else if (Util.isConvolutional(l)) {
		    lc.addConnectionCalculator(l, new ConnectionCalculatorConv());
		} else {
		    lc.addConnectionCalculator(l, new ConnectionCalculatorFullyConnected());
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return lc;
    }

    public static LayerCalculatorImpl lcSigmoid(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	for (Layer l : nn.getLayers()) {
	    if (!Util.isBias(l)) {
		if (outputCC != null && nn.getOutputLayer() == l) {
		    lc.addConnectionCalculator(l, outputCC);
		} else if (Util.isConvolutional(l)) {
		    lc.addConnectionCalculator(l, new AparapiConv2DSigmoid());
		} else if (!Util.isSubsampling(l)) {
		    lc.addConnectionCalculator(l, new AparapiSigmoid());
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return lc;
    }

    public static LayerCalculatorImpl lcSoftRelu(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	for (Layer l : nn.getLayers()) {
	    if (!Util.isBias(l)) {
		if (nn.getOutputLayer() == l) {
		    if (outputCC != null) {
			lc.addConnectionCalculator(l, outputCC);
		    } else {
			AparapiSoftReLU c = new AparapiSoftReLU();
			c.addActivationFunction(new SoftmaxFunction());
			lc.addConnectionCalculator(l, c);
		    }
		} else if (Util.isConvolutional(l)) {
		    lc.addConnectionCalculator(l, new AparapiConv2DSoftReLU());
		} else {
		    lc.addConnectionCalculator(l, new AparapiSoftReLU());
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return lc;
    }

    public static LayerCalculatorImpl lcRelu(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	for (Layer l : nn.getLayers()) {
	    if (!Util.isBias(l)) {
		if (nn.getOutputLayer() == l) {
		    if (outputCC != null) {
			lc.addConnectionCalculator(l, outputCC);
		    } else {
			AparapiReLU c = new AparapiReLU();
			c.addActivationFunction(new SoftmaxFunction());
			lc.addConnectionCalculator(l, c);
		    }
		} else if (Util.isConvolutional(l)) {
		    lc.addConnectionCalculator(l, new AparapiConv2DReLU());
		} else {
		    lc.addConnectionCalculator(l, new AparapiReLU());
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return lc;
    }

    public static LayerCalculatorImpl lcTanh(NeuralNetworkImpl nn, ConnectionCalculator outputCC) {
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	for (Layer l : nn.getLayers()) {
	    if (!Util.isBias(l)) {
		if (outputCC != null && nn.getOutputLayer() == l) {
		    lc.addConnectionCalculator(l, outputCC);
		} else if (Util.isConvolutional(l)) {
		    lc.addConnectionCalculator(l, new AparapiConv2DTanh());
		} else {
		    lc.addConnectionCalculator(l, new AparapiTanh());
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return lc;
    }

    public static void lcMaxPooling(NeuralNetworkImpl nn) {
	if (nn.getLayerCalculator() instanceof LayerCalculatorImpl) {
	    LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
	    for (Layer l : nn.getLayers()) {
		if (Util.isSubsampling(l)) {
		    lc.addConnectionCalculator(l, new AparapiMaxPooling2D());
		}
	    }
	} else {
	    throw new IllegalArgumentException("LayerCalculator type not supported");
	}
    }
    
    public static void lcAveragePooling(NeuralNetworkImpl nn) {
	if (nn.getLayerCalculator() instanceof LayerCalculatorImpl) {
	    LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
	    for (Layer l : nn.getLayers()) {
		if (Util.isSubsampling(l)) {
		    lc.addConnectionCalculator(l, new AparapiAveragePooling2D());
		}
	    }
	} else {
	    throw new IllegalArgumentException("LayerCalculator type not supported");
	}
    }
    
    public static void lcStochasticPooling(NeuralNetworkImpl nn) {
	if (nn.getLayerCalculator() instanceof LayerCalculatorImpl) {
	    LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
	    for (Layer l : nn.getLayers()) {
		if (Util.isSubsampling(l)) {
		    lc.addConnectionCalculator(l, new AparapiStochasticPooling2D());
		}
	    }
	} else {
	    throw new IllegalArgumentException("LayerCalculator type not supported");
	}
    }

    public static NeuralNetworkImpl mlpSigmoid(int[] layers, boolean addBias) {
	NeuralNetworkImpl result = mlp(layers, addBias);
	result.setLayerCalculator(lcSigmoid(result, null));
	return result;
    }

    public static NeuralNetworkImpl mlpSoftRelu(int[] layers, boolean addBias, ConnectionCalculator outputCC) {
	NeuralNetworkImpl result = mlp(layers, addBias);
	result.setLayerCalculator(lcSoftRelu(result, outputCC));
	return result;
    }

    public static NeuralNetworkImpl mlpRelu(int[] layers, boolean addBias, ConnectionCalculator outputCC) {
	NeuralNetworkImpl result = mlp(layers, addBias);
	result.setLayerCalculator(lcRelu(result, outputCC));
	return result;
    }

    public static NeuralNetworkImpl mlpTanh(int[] layers, boolean addBias, ConnectionCalculator outputCC) {
	NeuralNetworkImpl result = mlp(layers, addBias);
	result.setLayerCalculator(lcTanh(result, outputCC));
	return result;
    }

    public static Autoencoder autoencoder(int visibleCount, int hiddenCount, boolean addBias) {
	return new Autoencoder(visibleCount, hiddenCount, addBias);
    }

    public static Autoencoder autoencoderSigmoid(int visibleCount, int hiddenCount, boolean addBias) {
	Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
	ae.setLayerCalculator(lcSigmoid(ae, null));
	return ae;
    }

    public static Autoencoder autoencoderSoftReLU(int visibleCount, int hiddenCount, boolean addBias, ConnectionCalculator outputCC) {
	Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
	ae.setLayerCalculator(lcSoftRelu(ae, outputCC));
	return ae;
    }
    
    public static Autoencoder autoencoderReLU(int visibleCount, int hiddenCount, boolean addBias, ConnectionCalculator outputCC) {
	Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
	ae.setLayerCalculator(lcRelu(ae, outputCC));
	return ae;
    }

    public static Autoencoder autoencoderTanh(int visibleCount, int hiddenCount, boolean addBias, ConnectionCalculator outputCC) {
	Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
	ae.setLayerCalculator(lcTanh(ae, outputCC));
	return ae;
    }

    public static RBM rbm(int visibleCount, int hiddenCount, boolean addBias) {
	return new RBM(visibleCount, hiddenCount, addBias, addBias);
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
	result.addLayer(new Layer());
	for (int i = 1; i < layers.length; i++) {
	    result.addLevel(new Layer(), layers[i - 1], layers[i], addBias);
	}

	return result;
    }

    public static DBN dbnSigmoid(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);
	result.setLayerCalculator(lcSigmoid(result, null));
	return result;
    }
    
    public static DBN dbnSoftReLU(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);
	result.setLayerCalculator(lcSoftRelu(result, null));
	return result;
    }
    
    public static DBN dbnReLU(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);
	result.setLayerCalculator(lcRelu(result, null));
	return result;
    }

    public static DBN dbnTanh(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);
	result.setLayerCalculator(lcTanh(result, null));
	return result;
    }

    public static StackedAutoencoder sae(int[] layers, boolean addBias) {
	if (layers == null || layers.length <= 1) {
	    throw new IllegalArgumentException("more than one layer is required");
	}

	StackedAutoencoder result = new StackedAutoencoder(new Layer());
	for (int i = 1; i < layers.length; i++) {
	    result.addLevel(new Layer(), layers[i - 1], layers[i], addBias);
	}

	return result;
    }

    public static StackedAutoencoder saeSigmoid(int[] layers, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);
	sae.setLayerCalculator(lcSigmoid(sae, null));
	return sae;
    }

    public static StackedAutoencoder saeSoftReLU(int[] layers, int hiddenCount, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);
	sae.setLayerCalculator(lcSoftRelu(sae, null));
	return sae;
    }

    public static StackedAutoencoder saeReLU(int[] layers, int hiddenCount, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);
	sae.setLayerCalculator(lcRelu(sae, null));
	return sae;
    }

    public static StackedAutoencoder saeTanh(int[] layers, int hiddenCount, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);
	sae.setLayerCalculator(lcTanh(sae, null));
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
