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

    private Set<Layer> calculatedLayers;
    private Map<Layer, Matrix> results;
    private Map<Layer, Matrix> intermediateResults;

    public RBMLayerCalculator() {
	super();
	calculatedLayers = new HashSet<>();
	results = new HashMap<>();
	intermediateResults = new HashMap<>();
    }

    public void gibbsSampling(RBM rbm, Matrix posPhaseVisible, Matrix posPhaseHidden, Matrix negPhaseVisible, Matrix negPhaseHidden, int samplingCount, boolean resetNetwork, boolean useIntermediateResults) {
	Matrix hidden, visible;
	visible = getLayerResult(rbm.getVisibleLayer(), posPhaseVisible, useIntermediateResults);
	calculateHiddenLayer(rbm, visible, posPhaseHidden);

	if (resetNetwork) {
	    System.arraycopy(posPhaseHidden.getElements(), 0, negPhaseHidden.getElements(), 0, negPhaseHidden.getElements().length);
	}

	// Gibbs sampling
	for (int i = 1; i <= samplingCount; i++) {
	    hidden = getLayerResult(rbm.getHiddenLayer(), negPhaseHidden, useIntermediateResults);
	    calculateVisibleLayer(rbm, negPhaseVisible, hidden);

	    visible = getLayerResult(rbm.getVisibleLayer(), negPhaseVisible, useIntermediateResults);
	    calculateHiddenLayer(rbm, visible, negPhaseHidden);
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

    private Matrix getLayerResult(Layer layer, Matrix realResult, boolean useIntermediateResults) {
	Matrix result = realResult;
	if (useIntermediateResults) {
	    result = intermediateResults.get(layer);
	    if (result == null) {
		intermediateResults.put(layer, result = new Matrix(realResult));
	    }

	    System.arraycopy(realResult.getElements(), 0, result.getElements(), 0, result.getElements().length);
	}

	return result;
    }
}
