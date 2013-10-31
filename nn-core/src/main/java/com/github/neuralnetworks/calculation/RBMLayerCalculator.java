package com.github.neuralnetworks.calculation;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.RBM;

public class RBMLayerCalculator extends LayerCalculatorImpl {

    private static final long serialVersionUID = -7524966192939615856L;

    private RBM rbm;
    private ConnectionCalculator connectionCalculator;
    private Set<Layer> calculatedLayers;
    private Map<Layer, Matrix> results;

    public RBMLayerCalculator(RBM rbm) {
	super();
	this.rbm = rbm;
	calculatedLayers = new HashSet<>();
	results = new HashMap<>();
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

	calculate(calculatedLayers, results, visibleLayer);
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

	calculate(calculatedLayers, results, hiddenLayer);
    }

    @Override
    protected ConnectionCalculator getConnectionCalculator(Layer layer) {
	return connectionCalculator != null ? connectionCalculator : super.getConnectionCalculator(layer);
    }
}
