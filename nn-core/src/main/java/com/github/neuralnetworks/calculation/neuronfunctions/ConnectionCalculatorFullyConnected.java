package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.events.PropagationEvent;
import com.github.neuralnetworks.events.PropagationEventListener;
import com.github.neuralnetworks.util.Matrix;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.Tensor.TensorIterator;
import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.Util;

/**
 * Default implementation of Connection calculator for fully connected layers
 * Biases are also added After all the input functions are calculated there is a
 * list of activation functions that can be applied to the result This class
 * differs from LayerCalculatorImpl in the fact that LayerCalculatorImpl
 * traverses the graph of layers, where ConnectionCalculatorImpl only deals with
 * the connections passed as parameter
 * 
 * !!! Important !!! The results of the calculations are represented as tensors
 * (Tensor). This is done, because it is assumed that implementations will
 * provide a way for calculating many input results at once. Each column of the
 * matrix represents a single input. For example if the network is trained to
 * classify MNIST images, each column of the input matrix will represent single
 * MNIST image.
 */
public class ConnectionCalculatorFullyConnected implements ConnectionCalculator, PropagationEventListener {

    private static final long serialVersionUID = -5405654469496055017L;

    protected Set<ConnectionCalculator> inputFunctions;
    protected int miniBatchSize;

    /**
     * Activation functions that are executed before the transfer function
     */
    protected List<TensorFunction> preTransferFunctions;

    /**
     * Activation functions that are called after the transfer function
     */
    protected List<TensorFunction> activationFunctions;

    public ConnectionCalculatorFullyConnected() {
	super();
	inputFunctions = new HashSet<>();
    }

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	if (connections.size() > 0) {
	    List<Connections> notBias = new ArrayList<>();
	    Connections bias = null;

	    for (Connections c : connections) {
		// bias layer scenarios
		if (Util.isBias(c.getInputLayer())) {
		    bias = c;
		} else {
		    notBias.add(c);
		}
	    }

	    if (notBias.size() > 0) {
		if (preTransferFunctions != null && preTransferFunctions.size() > 0) {
		    preTransferFunctions.forEach(f -> notBias.stream().filter(c -> !Util.isBias(c.getInputLayer())).forEach(c -> f.value(valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c))));
		}

		calculateBias(bias, valuesProvider);

		getConnectionCalculator(notBias, valuesProvider, targetLayer).calculate(notBias, valuesProvider, targetLayer);

		if (activationFunctions != null) {
		    activationFunctions.forEach(f -> f.value(valuesProvider.getValues(targetLayer, notBias)));
		}
	    }
	}
    }

    @Override
    public void handleEvent(PropagationEvent event) {
	if (preTransferFunctions != null) {
	    preTransferFunctions.stream().filter(f -> f instanceof PropagationEventListener).forEach(f -> ((PropagationEventListener) f).handleEvent(event));
	}

	if (activationFunctions != null) {
	    activationFunctions.stream().filter(f -> f instanceof PropagationEventListener).forEach(f -> ((PropagationEventListener) f).handleEvent(event));
	}
    }

    public void addPreTransferFunction(TensorFunction function) {
	if (preTransferFunctions == null) {
	    preTransferFunctions = new UniqueList<>();
	}

	preTransferFunctions.add(function);
    }

    public void removePreTransfer(TensorFunction function) {
	if (preTransferFunctions != null) {
	    preTransferFunctions.remove(function);
	}
    }

    public void addActivationFunction(TensorFunction activationFunction) {
	if (activationFunctions == null) {
	    activationFunctions = new UniqueList<>();
	}

	activationFunctions.add(activationFunction);
    }

    public void removeActivationFunction(TensorFunction activationFunction) {
	if (activationFunctions != null) {
	    activationFunctions.remove(activationFunction);
	}
    }

    protected void calculateBias(Connections bias, ValuesProvider valuesProvider) {
	if (bias != null) {
	    Tensor biasValue = valuesProvider.getValues(bias.getInputLayer(), bias);
	    if (biasValue.get(new int[biasValue.getDimensions().length]) == 0) {
		biasValue.forEach(i -> biasValue.getElements()[i] = 1);
	    }

	    Matrix weights = ((FullyConnected) bias).getWeights();;
	    Matrix output = valuesProvider.getValues(bias.getOutputLayer(), bias);
	    TensorIterator it = output.iterator();

	    while (it.hasNext()) {
		int i = it.next();
		output.getElements()[i] = weights.get(it.getCurrentPosition()[0], 0);
	    }
	}
    }

    protected ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	return new AparapiWeightedSum(inputConnections, valuesProvider, targetLayer);
    }

    private ConnectionCalculator getConnectionCalculator(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	ConnectionCalculator result = inputFunctions.stream().filter(c -> {
	    return !(c instanceof AparapiWeightedSum) || ((AparapiWeightedSum) c).accept(connections, valuesProvider, targetLayer);
	}).findFirst().orElse(createInputFunction(connections, valuesProvider, targetLayer));
	inputFunctions.add(result);

	return result;
    }
}
