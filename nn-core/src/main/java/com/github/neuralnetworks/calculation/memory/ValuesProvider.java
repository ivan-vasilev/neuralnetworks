package com.github.neuralnetworks.calculation.memory;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.util.Tensor;

/**
 * Provides Tensor instances for the layers of the network. It ensures that the
 * instances are reused.
 */
public class ValuesProvider implements Serializable {

    private static final long serialVersionUID = 1L;

    private int miniBatchSize;
    private Map<Layer, Set<Tensor>> values;

    public ValuesProvider() {
	super();
	values = new HashMap<>();
    }

    /**
     * @return Tensor for connections. The connections must have a common layer and they must have the same dimensions.
     */
    public <T extends Tensor> T getValues(Layer targetLayer, Collection<Connections> connections) {
	return getValues(targetLayer, getDataDimensions(targetLayer, connections));
    }

    /**
     * @return Tensor for connections. The connections must have a common layer and they must have the same dimensions.
     */
    public <T extends Tensor> T getValues(Layer targetLayer, Connections c) {
	return getValues(targetLayer, Arrays.asList(new Connections[] {c}));
    }

    /**
     * @return Tensor for layer. Works only in the case when the layer has only one associated tensor.
     */
    public <T extends Tensor> T getValues(Layer targetLayer) {
	return getValues(targetLayer, targetLayer.getConnections());
    }

    /**
     * Get values for layer based on provided dimensions (excluding the minibatch size dimension)
     * @param targetLayer
     * @param unitCount
     * @return
     */
    @SuppressWarnings("unchecked")
    public <T extends Tensor> T getValues(Layer targetLayer, int... dimensions) {
	if (dimensions == null || dimensions.length == 0) {
	    throw new IllegalArgumentException("No dimensions provided");
	}

	if (!values.containsKey(targetLayer)) {
	    createValues(values);
	}

	return (T) values.get(targetLayer).stream().filter(t -> Arrays.equals(t.getDimensions(), dimensions)).findFirst().orElse(null);
    }

    public void addValues(Layer l, Tensor t) {
	Set<Tensor> set = values.get(l);
	if (set == null) {
	    values.put(l, set = new HashSet<Tensor>());
	} else {
	    set.removeIf(o -> o.getSize() == t.getSize());
	}

	setMiniBatchSize(t.getDimensions()[t.getDimensions().length - 1]);
	set.add(t);
    }

    public int getMiniBatchSize() {
	if (miniBatchSize == 0) {
	    values.values().forEach(v -> v.stream().filter(t -> miniBatchSize < t.getDimensions()[t.getDimensions().length - 1]).forEach(t -> miniBatchSize = t.getDimensions()[t.getDimensions().length - 1]));
	}

	return miniBatchSize;
    }

    public void setMiniBatchSize(int miniBatchSize) {
	this.miniBatchSize = miniBatchSize;
    }

    protected void createValues(Map<Layer, Set<Tensor>> values) {
    }

    private int[] getDataDimensions(Layer targetLayer, Collection<Connections> connections) {
	int[] result = null;
	boolean hasFullyConnected = false, hasSubsampling = false, hasConvolutional = false;
	for (Connections c : connections) {
	    if (c instanceof FullyConnected) {
		hasFullyConnected = true;
	    } else if (c instanceof Conv2DConnection) {
		hasConvolutional = true;
	    } else if (c instanceof Subsampling2DConnection) {
		hasSubsampling = true;
	    }
	}

	if (hasFullyConnected && (hasSubsampling || hasConvolutional)) {
	    throw new IllegalArgumentException("Cannot have fully connected and subsampling connections");
	}

	if (hasFullyConnected) {
	    result = new int[] {targetLayer.getUnitCount(connections), getMiniBatchSize() };
	} else if (hasSubsampling) {
	    Subsampling2DConnection c = (Subsampling2DConnection) connections.iterator().next();
	    if (c.getOutputLayer() == targetLayer) {
		result = new int[] { c.getFilters(), c.getOutputFeatureMapRows(), c.getOutputFeatureMapColumns(), getMiniBatchSize() };
	    } else if (c.getInputLayer() == targetLayer) {
		result = new int[] { c.getFilters(), c.getInputFeatureMapRows(), c.getInputFeatureMapColumns(), getMiniBatchSize() };
	    }
	} else if (hasConvolutional) {
	    Conv2DConnection c = (Conv2DConnection) connections.iterator().next();
	    if (c.getOutputLayer() == targetLayer) {
		result = new int[] { c.getOutputFilters(), c.getOutputFeatureMapRows(), c.getOutputFeatureMapColumns(), getMiniBatchSize() };
	    } else if (c.getInputLayer() == targetLayer) {
		result = new int[] { c.getInputFilters(), c.getInputFeatureMapRows(), c.getInputFeatureMapColumns(), getMiniBatchSize() };
	    }
	}

	return result;
    }

    protected Map<Layer, Set<int[]>> getLayerDimensions(NeuralNetwork neuralNetwork) {
	Map<Layer, Set<int[]>> result = new HashMap<>();

	for (Connections c : neuralNetwork.getConnections()) {
	    Set<int[]> din = result.get(c.getInputLayer());
	    if (din == null) {
		result.put(c.getInputLayer(), din = new HashSet<>());
	    }

	    Set<int[]> dout = result.get(c.getOutputLayer());
	    if (dout == null) {
		result.put(c.getOutputLayer(), dout = new HashSet<>());
	    }

	    if (c instanceof FullyConnected) {
		FullyConnected fc = (FullyConnected) c;
		int[] inputDim = new int[] { fc.getInputUnitCount(), getMiniBatchSize() };
		int[] outputDim = new int[] { fc.getOutputUnitCount(), getMiniBatchSize() };
		inputDim[0] = fc.getInputUnitCount();
		outputDim[0] = fc.getOutputUnitCount();

		if (!din.stream().anyMatch(i -> Arrays.equals(i, inputDim))) {
		    din.add(inputDim);
		}
		if (!dout.stream().anyMatch(i -> Arrays.equals(i, outputDim))) {
		    dout.add(outputDim);
		}
	    } else if (c instanceof Conv2DConnection) {
		Conv2DConnection cc = (Conv2DConnection) c;
		int[] inputDim = new int[] { cc.getOutputFilters(), cc.getOutputFeatureMapRows(), cc.getOutputFeatureMapColumns(), getMiniBatchSize() };
		int[] outputDim = new int[] { cc.getInputFilters(), cc.getInputFeatureMapRows(), cc.getInputFeatureMapColumns(), getMiniBatchSize() };
		if (!din.stream().anyMatch(i -> Arrays.equals(i, inputDim))) {
		    din.add(inputDim);
		}
		if (!dout.stream().anyMatch(i -> Arrays.equals(i, outputDim))) {
		    dout.add(outputDim);
		}
	    } else if (c instanceof Subsampling2DConnection) {
		Subsampling2DConnection cc = (Subsampling2DConnection) c;
		int[] inputDim = new int[] { cc.getFilters(), cc.getOutputFeatureMapRows(), cc.getOutputFeatureMapColumns(), getMiniBatchSize() };
		int[] outputDim = new int[] { cc.getFilters(), cc.getInputFeatureMapRows(), cc.getInputFeatureMapColumns(), getMiniBatchSize() };
		if (!din.stream().anyMatch(i -> Arrays.equals(i, inputDim))) {
		    din.add(inputDim);
		}
		if (!dout.stream().anyMatch(i -> Arrays.equals(i, outputDim))) {
		    dout.add(outputDim);
		}
	    }
	}

	return result;
    }
}
