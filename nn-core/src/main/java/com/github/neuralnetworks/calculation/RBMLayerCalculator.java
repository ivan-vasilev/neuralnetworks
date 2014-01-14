package com.github.neuralnetworks.calculation;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.RBM;

/**
 * Implementation of LayerCalculatorImpl for RBMs
 * Contains some helper methods like calculateVisibleLayer and calculateHiddenLayer and also takes gibbs sampling into account
 */
public class RBMLayerCalculator extends LayerCalculatorImpl {

    private static final long serialVersionUID = -7524966192939615856L;

    protected RBM rbm;
    protected ConnectionCalculator connectionCalculator;
    protected Set<Layer> calculatedLayers;
    protected Map<Layer, Matrix> results;

    public RBMLayerCalculator(RBM rbm) {
	super();
	this.rbm = rbm;
	calculatedLayers = new HashSet<>();
	results = new HashMap<>();
    }

    /* (non-Javadoc)
     * @see com.github.neuralnetworks.calculation.LayerCalculatorImpl#calculate(java.util.Set, java.util.Map, com.github.neuralnetworks.architecture.Layer)
     * takes into account the gibbs sampling - if the target "layer" is the input layer then the hidden layer is calculated first and then the visible
     */
    @Override
    public void calculate(Set<Layer> calculatedLayers, Map<Layer, Matrix> results, Layer layer) {
	Layer visibleLayer = rbm.getVisibleLayer();
	Layer hiddenLayer = rbm.getHiddenLayer();

	// gibbs sampling first
	if ((layer == visibleLayer || layer == rbm.getDataOutputLayer()) && layer != hiddenLayer && calculatedLayers.contains(visibleLayer)) {
	    super.calculate(calculatedLayers, results, hiddenLayer);
	    calculatedLayers.clear();
	    calculatedLayers.add(hiddenLayer);
	    super.calculate(calculatedLayers, results, visibleLayer);
	} else {
	    super.calculate(calculatedLayers, results, layer);
	}
    }

    public void calculateVisibleLayer(Matrix visibleLayerResults, Matrix hiddenLayerResults) {
	this.calculateVisibleLayer(visibleLayerResults, hiddenLayerResults, null);
    }

    public void calculateHiddenLayer(Matrix visibleLayerResults, Matrix hiddenLayerResults) {
	this.calculateHiddenLayer(visibleLayerResults, hiddenLayerResults, null);
    }

    public void calculateVisibleLayer(Matrix visibleLayerResults, Matrix hiddenLayerResults, ConnectionCalculator connectionCalculator) {
	this.connectionCalculator = connectionCalculator;

	Layer visibleLayer = rbm.getVisibleLayer();
	Layer hiddenLayer = rbm.getHiddenLayer();

	calculatedLayers.clear();
	calculatedLayers.add(hiddenLayer);

	results.clear();
	results.put(visibleLayer, visibleLayerResults);
	results.put(hiddenLayer, hiddenLayerResults);

	super.calculate(calculatedLayers, results, visibleLayer);
    }

    public void calculateHiddenLayer(Matrix visibleLayerResults, Matrix hiddenLayerResults, ConnectionCalculator connectionCalculator) {
	this.connectionCalculator = connectionCalculator;

	Layer visibleLayer = rbm.getVisibleLayer();
	Layer hiddenLayer = rbm.getHiddenLayer();

	calculatedLayers.clear();
	calculatedLayers.add(visibleLayer);

	results.clear();
	results.put(visibleLayer, visibleLayerResults);
	results.put(hiddenLayer, hiddenLayerResults);

	super.calculate(calculatedLayers, results, hiddenLayer);
    }

    @Override
    public ConnectionCalculator getConnectionCalculator(Layer layer) {
	return connectionCalculator != null ? connectionCalculator : super.getConnectionCalculator(layer);
    }
}
