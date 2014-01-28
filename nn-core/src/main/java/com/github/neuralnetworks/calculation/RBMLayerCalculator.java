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

    public RBMLayerCalculator() {
	super();
	calculatedLayers = new HashSet<>();
	results = new HashMap<>();
    }

    public void gibbsSampling(RBM rbm, Matrix posPhaseVisible, Matrix posPhaseHidden, Matrix negPhaseVisible, Matrix negPhaseHidden, int samplingCount, boolean resetNetwork) {
	calculateHiddenLayer(rbm, posPhaseVisible, posPhaseHidden);

	if (resetNetwork) {
	    System.arraycopy(posPhaseHidden.getElements(), 0, negPhaseHidden.getElements(), 0, negPhaseHidden.getElements().length);
	}

	// Gibbs sampling
	for (int i = 1; i <= samplingCount; i++) {
	    calculateVisibleLayer(rbm, negPhaseVisible, negPhaseHidden);
	    calculateHiddenLayer(rbm, negPhaseVisible, negPhaseHidden);
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
