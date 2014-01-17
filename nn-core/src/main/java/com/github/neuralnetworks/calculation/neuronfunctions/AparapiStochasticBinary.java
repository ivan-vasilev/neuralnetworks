package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.training.random.RandomInitializer;

/**
 * Stochastic binary activation function (for RBMs for example)
 */
public class AparapiStochasticBinary extends ConnectionCalculatorFullyConnected {

    private static final long serialVersionUID = 5869298546838843306L;

    protected RandomInitializer randominitializer;

    public AparapiStochasticBinary(RandomInitializer randominitializer) {
	this.randominitializer = randominitializer;
    }

    @Override
    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Integer> inputConnections, Layer targetLayer) {
	return new AparapiStochasticBinaryFunction(inputConnections, miniBatchSize, targetLayer, randominitializer);
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

	public AparapiStochasticBinaryFunction(SortedMap<GraphConnections, Integer> inputConnections, int miniBatchSize, Layer targetLayer, RandomInitializer randomInitializer) {
	    super(inputConnections, miniBatchSize, targetLayer);
	    this.randomInitializer = randomInitializer;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.github.neuralnetworks.calculation.neuronfunctions.
	 * AparapiWeightedSumByRows#init(java.util.SortedMap,
	 * com.github.neuralnetworks.architecture.Matrix,
	 * com.github.neuralnetworks.architecture.Layer) Unfortunately there
	 * isn't yet random implementation that works for Aparapi, so this step
	 * is done sequential
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
	protected void after() {
	    int mb = miniBatchSize;
	    int outputId = getGlobalId() * mb;
	    
	    for (int i = 0; i < mb; i++) {
		if (random[outputId + i] < 1 / (1 + exp(-output[outputId + i]))) {
		    output[outputId + i] = 1;
		} else {
		    output[outputId + i] = 0;
		}
	    }
	}
    }
}
