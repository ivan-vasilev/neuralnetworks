package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;
import com.github.neuralnetworks.training.random.RandomInitializer;

/**
 * Stochastic binary activation function (for RBMs for example)
 */
public class AparapiStochasticBinary extends ConnectionCalculatorImpl {

    private static final long serialVersionUID = 5869298546838843306L;

    public AparapiStochasticBinary(RandomInitializer randominitializer) {
	super(new AparapiStochasticBinaryFunction(randominitializer));
    }

    public static class AparapiStochasticBinaryFunction extends AparapiWeightedSum {

	private static final long serialVersionUID = -9125510037725731152L;

	/**
	 * random values for the stochastic activation of neurons
	 */
	private float[] random;

	/**
	 * random initializer
	 */
	private RandomInitializer randomInitializer;

	public AparapiStochasticBinaryFunction(RandomInitializer randominitializer) {
	    super();
	    randomInitializer = randominitializer;
	}

	/* (non-Javadoc)
	 * @see com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSumByRows#init(java.util.SortedMap, com.github.neuralnetworks.architecture.Matrix, com.github.neuralnetworks.architecture.Layer)
	 * Unfortunately there isn't yet random implementation that works for Aparapi, so this step is done sequential
	 */
	@Override
	protected void init(SortedMap<GraphConnections, Matrix> input, Matrix outputMatrix, Layer targetLayer) {
	    super.init(input, outputMatrix, targetLayer);
	    if (random == null || random.length != outputMatrix.getElements().length) {
		random = new float[outputMatrix.getElements().length];
	    }

	    randomInitializer.initialize(random);
	}

	@Override
	protected void after(float value, int row, int column) {
	    int outputId = outputIndex(row, column);
	    if (random[outputId] < 1 / (1 + exp(-value))) {
		output[outputId] = 1;
	    } else {
		output[outputId] = 0;
	    }
	}
    }
}
