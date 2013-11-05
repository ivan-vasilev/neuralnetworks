package com.github.neuralnetworks.calculation;

import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.SupervisedRBM;
import com.github.neuralnetworks.util.Util;

public class SupervisedRBMLayerCalculator extends RBMLayerCalculator {

    private static final long serialVersionUID = -8397884896433147900L;

    private Matrix combinedInput;
    private Matrix realInput;
    private Matrix targetOutput;

    public SupervisedRBMLayerCalculator(SupervisedRBM rbm) {
	super(rbm);
    }

    @Override
    public void calculate(Set<Layer> calculatedLayers, Map<Layer, Matrix> results, Layer layer) {
	combineInput(results);
	super.calculate(calculatedLayers, results, layer);
	splitResults(results);
    }

    @Override
    public void calculateVisibleLayer(Matrix visibleLayerResults, Matrix hiddenLayerResults, ConnectionCalculator connectionCalculator) {
	results.clear();
	results.put(rbm.getVisibleLayer(), visibleLayerResults);
	results.put(rbm.getHiddenLayer(), hiddenLayerResults);
	combineInput(results);
	super.calculateVisibleLayer(visibleLayerResults, hiddenLayerResults, connectionCalculator);
	splitResults(results);
    }

    @Override
    public void calculateHiddenLayer(Matrix visibleLayerResults, Matrix hiddenLayerResults, ConnectionCalculator connectionCalculator) {
	results.clear();
	results.put(rbm.getVisibleLayer(), visibleLayerResults);
	results.put(rbm.getHiddenLayer(), hiddenLayerResults);
	combineInput(results);
	super.calculateHiddenLayer(visibleLayerResults, hiddenLayerResults, connectionCalculator);
	splitResults(results);
    }

    protected void combineInput(Map<Layer, Matrix> results) {
	Layer visibleLayer = rbm.getVisibleLayer();
	realInput = results.get(visibleLayer);
	if (realInput != null && realInput.getRows() != visibleLayer.getNeuronCount()) {
	    if (combinedInput == null || combinedInput.getRows() != visibleLayer.getNeuronCount() || combinedInput.getColumns() != realInput.getColumns()) {
		combinedInput = new Matrix(visibleLayer.getNeuronCount(), realInput.getColumns());
	    } else {
		Util.fillArray(combinedInput.getElements(), 0);
	    }

	    System.arraycopy(realInput.getElements(), 0, combinedInput.getElements(), 0, realInput.getElements().length);
	    targetOutput = results.get(rbm.getDataOutputLayer());
	    if (targetOutput != realInput && targetOutput != null) {
		results.remove(rbm.getDataOutputLayer());
		System.arraycopy(targetOutput.getElements(), 0, combinedInput.getElements(), realInput.getElements().length, targetOutput.getElements().length);
	    } else {
		targetOutput = null;
	    }

	    results.put(visibleLayer, combinedInput);
	} else {
	    combinedInput = realInput;
	}
    }

    protected void splitResults(Map<Layer, Matrix> results) {
	Layer visibleLayer = rbm.getVisibleLayer();
	if (realInput != null && realInput.getRows() != visibleLayer.getNeuronCount()) {
	    System.arraycopy(combinedInput.getElements(), 0, realInput.getElements(), 0, realInput.getElements().length);

	    Layer dataOutputLayer = rbm.getDataOutputLayer();
	    if (dataOutputLayer != null) {
		if (targetOutput == null || targetOutput.getColumns() != realInput.getColumns() || targetOutput.getRows() != dataOutputLayer.getNeuronCount()) {
		    targetOutput = new Matrix(dataOutputLayer.getNeuronCount(), realInput.getColumns());
		}

		System.arraycopy(combinedInput.getElements(), realInput.getElements().length, targetOutput.getElements(), 0, targetOutput.getElements().length);
		if (dataOutputLayer.getConnectionCalculator() != null) {
		    dataOutputLayer.getConnectionCalculator().calculate(null, targetOutput, null);
		}

		results.put(dataOutputLayer, targetOutput);
	    }

	    results.put(visibleLayer, realInput);
	}
    }
}
 