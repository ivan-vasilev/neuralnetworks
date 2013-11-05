package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;

public class ThresholdConnectionCalculator implements ConnectionCalculator {

    private static final long serialVersionUID = 64378606439160209L;

    private float threshold;

    public ThresholdConnectionCalculator(float threshold) {
	super();
	this.threshold = threshold;
    }

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	float[] elements = output.getElements();
	for (int i = 0; i < elements.length; i++) {
	    elements[i] = elements[i] >= threshold ? 1 : 0;
	}
    }
}
