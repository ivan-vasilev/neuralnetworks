package com.github.neuralnetworks.calculation;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.RBM;

/**
 * Implementation of LayerCalculatorImpl for RBMs
 * Contains some helper methods like calculateVisibleLayer and calculateHiddenLayer and also takes gibbs sampling into account
 */
public class RBMLayerCalculator extends LayerCalculatorImpl {

    private static final long serialVersionUID = -7524966192939615856L;

    protected Set<Layer> calculatedLayers;
    protected Map<Layer, Matrix> results;

    public RBMLayerCalculator() {
	super();
	calculatedLayers = new HashSet<>();
	results = new HashMap<>();
    }

    /* (non-Javadoc)
     * @see com.github.neuralnetworks.calculation.LayerCalculatorImpl#calculate(java.util.Set, java.util.Map, com.github.neuralnetworks.architecture.Layer)
     * takes into account the gibbs sampling - if the target "layer" is the input layer then the hidden layer is calculated first and then the visible
     */
    @Override
    public void calculate(NeuralNetwork neuralNetwork, Layer layer, Set<Layer> calculatedLayers, Map<Layer, Matrix> results) {
	this.results = results;

	RBM rbm = (RBM) neuralNetwork;
	Layer visibleLayer = rbm.getVisibleLayer();
	Layer hiddenLayer = rbm.getHiddenLayer();

	// gibbs sampling first
	if ((layer == visibleLayer || layer == rbm.getOutputLayer()) && layer != hiddenLayer && calculatedLayers.contains(visibleLayer)) {
	    super.calculate(neuralNetwork, hiddenLayer, calculatedLayers, results);
	    calculatedLayers.clear();
	    calculatedLayers.add(hiddenLayer);
	    super.calculate(neuralNetwork, visibleLayer, calculatedLayers, results);
	} else {
	    super.calculate(neuralNetwork, layer, calculatedLayers, results);
	}
    }

    public void calculateVisibleLayer(RBM rbm, Matrix visibleLayerResults, Matrix hiddenLayerResults) {
	Layer visibleLayer = rbm.getVisibleLayer();
	Layer hiddenLayer = rbm.getHiddenLayer();

	calculatedLayers.clear();
	calculatedLayers.add(hiddenLayer);

	results.put(visibleLayer, visibleLayerResults);
	results.put(hiddenLayer, hiddenLayerResults);

	super.calculate(rbm, visibleLayer, calculatedLayers, results);
    }

    public void calculateHiddenLayer(RBM rbm, Matrix visibleLayerResults, Matrix hiddenLayerResults) {
	Layer visibleLayer = rbm.getVisibleLayer();
	Layer hiddenLayer = rbm.getHiddenLayer();

	calculatedLayers.clear();
	calculatedLayers.add(visibleLayer);

	results.put(visibleLayer, visibleLayerResults);
	results.put(hiddenLayer, hiddenLayerResults);

	super.calculate(rbm, hiddenLayer, calculatedLayers, results);
    }
}
