package com.github.neuralnetworks.training.backpropagation;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiFullyConnected;
import com.github.neuralnetworks.calculation.neuronfunctions.MaxoutWinners;
import com.github.neuralnetworks.util.Matrix;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.TensorFactory;

public class BackpropagationMaxout extends BackPropagationConnectionCalculatorImpl {

    private static final long serialVersionUID = 1L;

    public BackpropagationMaxout(Properties properties) {
	super(properties);
    }

    @Override
    protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider, ValuesProvider activations, Layer targetLayer) {
	for (Connections c : inputConnections) {
	    connectionCalculators.put(c, new AparapiBackpropMaxout(c, valuesProvider, activations, Arrays.asList(getWeightUpdates().get(c)), getLearningRate(), getMomentum(), getL1weightDecay(), getL2weightDecay()));
	}
    }

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	targetLayer = connections.get(0).getOutputLayer();
	for (Connections c : connections) {
	    if (targetLayer != c.getOutputLayer()) {
		throw new IllegalArgumentException("No common target layer");
	    }
	}

	super.calculate(connections, valuesProvider, targetLayer);
    }

    public static class AparapiBackpropMaxout extends AparapiFullyConnected implements BackPropagationConnectionCalculator {

	private static final long serialVersionUID = 1L;

	/**
	 * Activation of the output layer from the feedforward phase
	 */
	@Constant
	protected float[] ffActivation;
	protected final int activationStartPosition;
	protected final int activationRowStep;
	protected final int activationColumnStep;

	/**
	 * Weight updates array
	 */
	protected final float[] weightUpdates;

	protected float learningRate;
	protected final float momentum;
	protected final float l1weightDecay;
	protected final float l2weightDecay;

	private final int[] winnersStartPositions;
	private final int[] maxoutWinners;

	public AparapiBackpropMaxout(Connections inputConnection, ValuesProvider valuesProvider, ValuesProvider activations, List<Tensor> weightUpdates, float learningRate, float momentum, float l1weightDecay, float l2weightDecay) {
	    super(Arrays.asList(new Connections[] {inputConnection}), valuesProvider, inputConnection.getOutputLayer());

	    Matrix m = TensorFactory.tensor(inputConnection.getInputLayer(), inputConnection, activations);
	    this.ffActivation = m.getElements();
	    this.activationStartPosition = m.getStartIndex();
	    this.activationRowStep = m.getRowElementsDistance();
	    this.activationColumnStep = m.getColumnElementsDistance();

	    this.learningRate = momentum;
	    this.momentum = momentum;
	    this.l1weightDecay = l1weightDecay;
	    this.l2weightDecay = l2weightDecay;
	    this.weightUpdates = weightUpdates.get(0).getElements();

	    this.winnersStartPositions = MaxoutWinners.getInstance().getStartPositions(Arrays.asList(new Connections[] {inputConnection}));
	    this.maxoutWinners = MaxoutWinners.getInstance().getWinners();
	}

	@Override
	public void run() {
	    int id = getGlobalId();

	    int maxoutId = 0, weightId = 0;
	    float weight = 0, weightUpdate = 0;

	    // each input example
	    for (int i = 0; i < miniBatchSize; i++) {
		// each connection (of the combined connections)
		for (int k = 0; k < series; k++) {
		    maxoutId = maxoutWinners[winnersStartPositions[k] + id * miniBatchSize + i];
		    weightId = weightStartPositions[k] + weightsInitialStep[k] * id + maxoutId * weightsStep[k];
		    weight = weights[weightId];

		    weightUpdate += output[outputStartPosition + id * outputRowStep + i * outputColumnStep] * ffActivation[activationStartPosition + maxoutId * activationRowStep + i * activationColumnStep];
		    weightUpdate = learningRate * weightUpdate + momentum * weightUpdates[weightId] - l1weightDecay * abs(weight) - l2weightDecay * weight * weight / 2;
		    weights[weightId] += weightUpdate;
		    weightUpdates[weightId] = weightUpdate;

		    input[activationStartPosition + maxoutId * activationRowStep + i * activationColumnStep] += output[outputStartPosition + id * outputRowStep + i * outputColumnStep];
		}
	    }
	}

	@Override
	public float getLearningRate() {
	    return learningRate;
	}

	@Override
	public void setLearningRate(float learningRate) {
	    this.learningRate = learningRate;
	}

	@Override
	public float getMomentum() {
	    return momentum;
	}

	@Override
	public void setMomentum(float momentum) {
	}

	@Override
	public float getL1weightDecay() {
	    return l1weightDecay;
	}

	@Override
	public void setL1weightDecay(float weightDecay) {
	}

	@Override
	public float getL2weightDecay() {
	    return l2weightDecay;
	}

	@Override
	public void setL2weightDecay(float l2weightDecay) {
	}

	@Override
	public ValuesProvider getActivations() {
	    return null;
	}

	@Override
	public void setActivations(ValuesProvider activations) {
	}
    }
}
