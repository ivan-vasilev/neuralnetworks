package com.github.neuralnetworks.architecture.types;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSoftReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiStochasticBinary;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiTanh;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;
import com.github.neuralnetworks.training.random.AparapiXORShiftInitializer;
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

    public static MultiLayerPerceptron mlpSigmoid(int[] layers, boolean addBias) {
	MultiLayerPerceptron result = mlp(layers, addBias);

	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	result.setLayerCalculator(lc);
	for (Layer l : result.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != result.getInputLayer()) {
		    lc.addConnectionCalculator(l, new AparapiSigmoid());
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return result;
    }
    
    public static MultiLayerPerceptron mlpSigmoid(int[] layers, boolean addBias, ConnectionCalculator outputCC) {
	MultiLayerPerceptron result = mlp(layers, addBias);
	
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	result.setLayerCalculator(lc);
	for (Layer l : result.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != result.getInputLayer()) {
		    if (outputCC != null && result.getOutputLayer() == l) {
			lc.addConnectionCalculator(l, outputCC);
		    } else {
			lc.addConnectionCalculator(l, new AparapiSigmoid());
		    }
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}
	
	return result;
    }

    public static MultiLayerPerceptron mlpSoftRelu(int[] layers, boolean addBias, ConnectionCalculator outputCC) {
	MultiLayerPerceptron result = mlp(layers, addBias);

	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	result.setLayerCalculator(lc);
	for (Layer l : result.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != result.getInputLayer()) {
		    if (outputCC != null && result.getOutputLayer() == l) {
			lc.addConnectionCalculator(l, outputCC);
		    } else {
			lc.addConnectionCalculator(l, new AparapiSoftReLU());
		    }
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return result;
    }

    public static MultiLayerPerceptron mlpRelu(int[] layers, boolean addBias, ConnectionCalculator outputCC) {
	MultiLayerPerceptron result = mlp(layers, addBias);
	
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	result.setLayerCalculator(lc);
	for (Layer l : result.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != result.getInputLayer()) {
		    if (outputCC != null && result.getOutputLayer() == l) {
			lc.addConnectionCalculator(l, outputCC);
		    } else {
			lc.addConnectionCalculator(l, new AparapiReLU());
		    }
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}
	
	return result;
    }

    public static MultiLayerPerceptron mlpTanh(int[] layers, boolean addBias, ConnectionCalculator outputCC) {
	MultiLayerPerceptron result = mlp(layers, addBias);

	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	result.setLayerCalculator(lc);
	for (Layer l : result.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != result.getInputLayer()) {
		    if (outputCC != null && result.getOutputLayer() == l) {
			lc.addConnectionCalculator(l, outputCC);
		    } else {
			lc.addConnectionCalculator(l, new AparapiTanh());
		    }
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return result;
    }

    public static Autoencoder autoencoderSigmoid(int visibleCount, int hiddenCount, boolean addBias) {
	Layer visible = new Layer(visibleCount);
	Layer hidden = new Layer(hiddenCount);
	Layer output = new Layer(visibleCount);
	Autoencoder ae = new Autoencoder(visible, hidden, output, addBias);

	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	ae.setLayerCalculator(lc);
	lc.addConnectionCalculator(hidden, new AparapiSigmoid());
	lc.addConnectionCalculator(output, new AparapiSigmoid());

	return ae;
    }

    public static Autoencoder autoencoderSoftReLU(int visibleCount, int hiddenCount, boolean addBias) {
	Layer visible = new Layer(visibleCount);
	Layer hidden = new Layer(hiddenCount);
	Layer output = new Layer(visibleCount);
	Autoencoder ae = new Autoencoder(visible, hidden, output, addBias);

	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	ae.setLayerCalculator(lc);
	lc.addConnectionCalculator(hidden, new AparapiSoftReLU());
	lc.addConnectionCalculator(output, new AparapiSoftReLU());
	if (addBias) {
	    Util.populateBiasLayers(lc, ae);
	}

	return ae;
    }
    
    public static Autoencoder autoencoderReLU(int visibleCount, int hiddenCount, boolean addBias) {
	Layer visible = new Layer(visibleCount);
	Layer hidden = new Layer(hiddenCount);
	Layer output = new Layer(visibleCount);
	Autoencoder ae = new Autoencoder(visible, hidden, output, addBias);
	
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	ae.setLayerCalculator(lc);
	lc.addConnectionCalculator(hidden, new AparapiReLU());
	lc.addConnectionCalculator(output, new AparapiReLU());
	if (addBias) {
	    Util.populateBiasLayers(lc, ae);
	}
	
	return ae;
    }

    public static Autoencoder autoencoderTanh(int visibleCount, int hiddenCount, boolean addBias) {
	Layer visible = new Layer(visibleCount);
	Layer hidden = new Layer(hiddenCount);
	Layer output = new Layer(visibleCount);
	Autoencoder ae = new Autoencoder(visible, hidden, output, addBias);

	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	ae.setLayerCalculator(lc);
	lc.addConnectionCalculator(hidden, new AparapiTanh());
	lc.addConnectionCalculator(output, new AparapiTanh());
	if (addBias) {
	    Util.populateBiasLayers(lc, ae);
	}

	return ae;
    }

    public static RBM rbmSigmoidSigmoid(int visibleCount, int hiddenCount, boolean addBias) {
	RBM rbm = new RBM(new Layer(visibleCount), new Layer(hiddenCount), addBias, addBias);
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiSigmoid());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiSigmoid());
	if (addBias) {
	    Util.populateBiasLayers(lc, rbm);
	}

	return rbm;
    }

    public static RBM rbmSoftReluSoftRelu(int visibleCount, int hiddenCount, boolean addBias) {
	RBM rbm = new RBM(new Layer(visibleCount), new Layer(hiddenCount), addBias, addBias);
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiSoftReLU());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiSoftReLU());
	if (addBias) {
	    Util.populateBiasLayers(lc, rbm);
	}

	return rbm;
    }
    
    public static RBM rbmReluRelu(int visibleCount, int hiddenCount, boolean addBias) {
	RBM rbm = new RBM(new Layer(visibleCount), new Layer(hiddenCount), addBias, addBias);
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiReLU());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiReLU());
	if (addBias) {
	    Util.populateBiasLayers(lc, rbm);
	}
	
	return rbm;
    }

    public static RBM rbmTanhTanh(int visibleCount, int hiddenCount, boolean addBias) {
	RBM rbm = new RBM(new Layer(visibleCount), new Layer(hiddenCount), addBias, addBias);
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiTanh());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiTanh());
	if (addBias) {
	    Util.populateBiasLayers(lc, rbm);
	}

	return rbm;
    }

    public static RBM rbmSigmoidBinary(int visibleCount, int hiddenCount, boolean addBias) {
	RBM rbm = new RBM(new Layer(visibleCount), new Layer(hiddenCount), addBias, addBias);
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiSigmoid());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiStochasticBinary(new AparapiXORShiftInitializer()));
	if (addBias) {
	    Util.populateBiasLayers(lc, rbm);
	}

	return rbm;
    }

    public static RBM rbmSoftReluBinary(int visibleCount, int hiddenCount, boolean addBias) {
	RBM rbm = new RBM(new Layer(visibleCount), new Layer(hiddenCount), addBias, addBias);
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiSoftReLU());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiStochasticBinary(new AparapiXORShiftInitializer()));

	return rbm;
    }
    
    public static RBM rbmReluBinary(int visibleCount, int hiddenCount, boolean addBias) {
	RBM rbm = new RBM(new Layer(visibleCount), new Layer(hiddenCount), addBias, addBias);
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiReLU());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiStochasticBinary(new AparapiXORShiftInitializer()));
	
	return rbm;
    }

    public static RBM rbmTanhBinary(int visibleCount, int hiddenCount, boolean addBias) {
	RBM rbm = new RBM(new Layer(visibleCount), new Layer(hiddenCount), addBias, addBias);
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	rbm.setLayerCalculator(lc);
	lc.addConnectionCalculator(rbm.getVisibleLayer(), new AparapiTanh());
	lc.addConnectionCalculator(rbm.getHiddenLayer(), new AparapiStochasticBinary(new AparapiXORShiftInitializer()));

	return rbm;
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

	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	result.setLayerCalculator(lc);
	for (Layer l : result.getLayers()) {
	    if (!(l instanceof BiasLayer) && l != result.getInputLayer()) {
		lc.addConnectionCalculator(l, new AparapiSigmoid());
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return result;
    }
    
    public static DBN dbnSoftReLU(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);

	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	result.setLayerCalculator(lc);
	for (Layer l : result.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != result.getInputLayer()) {
		    lc.addConnectionCalculator(l, new AparapiSoftReLU());
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return result;
    }
    
    public static DBN dbnReLU(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);
	
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	result.setLayerCalculator(lc);
	for (Layer l : result.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != result.getInputLayer()) {
		    lc.addConnectionCalculator(l, new AparapiReLU());
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}
	
	return result;
    }

    public static DBN dbnTanh(int[] layers, boolean addBias) {
	DBN result = dbn(layers, addBias);

	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	result.setLayerCalculator(lc);
	for (Layer l : result.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != result.getInputLayer()) {
		    lc.addConnectionCalculator(l, new AparapiTanh());
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

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

	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	sae.setLayerCalculator(lc);
	for (Layer l : sae.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != sae.getInputLayer()) {
		    lc.addConnectionCalculator(l, new AparapiSigmoid());
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return sae;
    }

    public static StackedAutoencoder saeSoftReLU(int[] layers, int hiddenCount, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);

	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	sae.setLayerCalculator(lc);
	for (Layer l : sae.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != sae.getInputLayer()) {
		    lc.addConnectionCalculator(l, new AparapiSoftReLU());
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return sae;
    }
    
    public static StackedAutoencoder saeReLU(int[] layers, int hiddenCount, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);
	
	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	sae.setLayerCalculator(lc);
	for (Layer l : sae.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != sae.getInputLayer()) {
		    lc.addConnectionCalculator(l, new AparapiReLU());
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}
	
	return sae;
    }
    
    public static StackedAutoencoder saeTanh(int[] layers, int hiddenCount, boolean addBias) {
	StackedAutoencoder sae = sae(layers, addBias);

	LayerCalculatorImpl lc = new LayerCalculatorImpl();
	sae.setLayerCalculator(lc);
	for (Layer l : sae.getLayers()) {
	    if (!(l instanceof BiasLayer)) {
		if (l != sae.getInputLayer()) {
		    lc.addConnectionCalculator(l, new AparapiTanh());
		}
	    } else {
		lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
	    }
	}

	return sae;
    }
}
