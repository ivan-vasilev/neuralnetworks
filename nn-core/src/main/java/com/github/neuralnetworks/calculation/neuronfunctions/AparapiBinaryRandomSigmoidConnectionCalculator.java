package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;
import com.github.neuralnetworks.training.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.RandomInitializer;

public class AparapiBinaryRandomSigmoidConnectionCalculator extends ConnectionCalculatorImpl {

    private static final long serialVersionUID = 5869298546838843306L;

    public AparapiBinaryRandomSigmoidConnectionCalculator() {
	super(new AparapiBinaryRandomSigmoidByRows(), new AparapiBinaryRandomSigmoidByColumns(), null);
    }

    public static class AparapiBinaryRandomSigmoidByRows extends AparapiWeightedSumByRows {

	private static final long serialVersionUID = -9125510037725731152L;

	private float[] random;
	private RandomInitializer randomInitializer;

	public AparapiBinaryRandomSigmoidByRows() {
	    super();
	    randomInitializer = new MersenneTwisterRandomInitializer();
	}

	@Override
	protected void init(SortedMap<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	    super.init(input, outputMatrix, targetLayer);
	    if (random == null || random.length != outputMatrix.getElements().length) {
		random = new float[outputMatrix.getElements().length];
	    }

	    randomInitializer.initialize(random);
	}

	@Override
	protected void after(int row, int column) {
	    if (random[outputBaseIndex(row, column)] < 1 / (1 + exp(-output[outputBaseIndex(row, column)]))) {
		output[outputBaseIndex(row, column)] = 1;
	    } else {
		output[outputBaseIndex(row, column)] = 0;
	    }
	}
    }

    public static class AparapiBinaryRandomSigmoidByColumns extends AparapiWeightedSumByColumns {

	private static final long serialVersionUID = -9125510037725731152L;

	private float[] random;
	private RandomInitializer randomInitializer;

	public AparapiBinaryRandomSigmoidByColumns() {
	    super();
	    randomInitializer = new MersenneTwisterRandomInitializer();
	}

	@Override
	protected void init(SortedMap<Connections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	    super.init(input, outputMatrix, targetLayer);
	    if (random == null || random.length != outputMatrix.getElements().length) {
		random = new float[outputMatrix.getElements().length];
	    }

	    randomInitializer.initialize(random);
	}

	@Override
	protected void after(int row, int column) {
	    int index = outputBaseIndex(row, column);
	    if (random[index] < 1 / (1 + exp(-output[index]))) {
		output[index] = 1;
	    } else {
		output[index] = 0;
	    }
	}
    }
}
