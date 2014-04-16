package com.github.neuralnetworks.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.util.Tensor.TensorIterator;

public class TensorFactory {

    @SuppressWarnings("unchecked")
    public static <T extends Tensor> T tensor(int... dimensions) {
	float[] elements = new float[IntStream.of(dimensions).reduce(1, (a, b) -> a * b)];
	int[][] dimensionsLimit = new int[2][dimensions.length];
	IntStream.range(0, dimensions.length).forEach(i -> dimensionsLimit[1][i] = dimensions[i] - 1);

	T result = null;
	if (dimensions.length == 2) {
	    result = (T) new Matrix(0, elements, dimensions, dimensionsLimit);
	} else {
	    result = (T) new Tensor(0, elements, dimensions, dimensionsLimit);
	}

	return result;
    }

    @SuppressWarnings("unchecked")
    public static <T extends Tensor> T tensor(float[] elements, int offset, int... dimensions) {
	int[][] dimensionsLimit = new int[2][dimensions.length];
	IntStream.range(0, dimensions.length).forEach(i -> dimensionsLimit[1][i] = dimensions[i] - 1);

	T result = null;
	if (dimensions.length == 2) {
	    result = (T) new Matrix(offset, elements, dimensions, dimensionsLimit);
	} else {
	    result = (T) new Tensor(offset, elements, dimensions, dimensionsLimit);
	}

	return result;
    }

    @SuppressWarnings("unchecked")
    public static <T extends Tensor> T tensor(Tensor parent, int[][] dimensionsLimit) {
	T result = null;

	long dimensions = IntStream.range(0, dimensionsLimit[0][0]).filter(i -> dimensionsLimit[0][i] != dimensionsLimit[1][i]).count();

	if (dimensions <= 2) {
	    result = (T) new Matrix(parent, dimensionsLimit);
	} else {
	    result = (T) new Tensor(parent, dimensionsLimit);
	}

	return result;
    }

    /**
     * @param copy
     * @return new tensor with the same dimensions
     */
    public static <T extends Tensor> T tensor(Tensor copy) {
	return tensor(copy.getDimensions());
    }

    /**
     * Create multiple combined tensors using shared elements array
     * @param dimensions - dimensions for each tensor
     * @return array of tensors
     */
    public static Tensor[] tensor(int[]... dimensions) {
	Tensor[] result = new Tensor[dimensions.length];
	float[] elements = new float[Arrays.stream(dimensions).map(d -> {
	    return IntStream.of(d).reduce(1, (a, b) -> a * b);
	}).reduce(0, (a, b) -> a + b)];

	for (int i = 0, offset = 0; i < dimensions.length; i++) {
	    int[] d = dimensions[i];
	    int[][] dimensionsLimit = new int[2][d.length];
	    IntStream.range(0, d.length).forEach(j -> dimensionsLimit[1][j] = d[j] - 1);
	    if (d.length == 2) {
		result[i] = new Matrix(offset, elements, d, dimensionsLimit);
	    } else {
		result[i] = new Tensor(offset, elements, d, dimensionsLimit);
	    }

	    offset += IntStream.of(d).reduce(1, (a, b) -> a * b);
	}

	return result;
    }

    /**
     * Simplified construction of matrix using values
     * @param elements
     * @return Matrix
     */
    public static Matrix matrix(float[][] elements) {
	Matrix result = tensor(elements[0].length, elements.length);
	IntStream.range(0, elements.length).forEach(i -> IntStream.range(0, elements[i].length).forEach(j -> {
	    result.set(elements[i][j], j, i);
	}));

	return result;
    }

    public static Matrix matrix(float[] elements, int columns) {
	return tensor(elements, 0, elements.length / columns, columns);
    }

    public static void fill(Tensor t, float value) {
	TensorIterator it = t.iterator();
	while (it.hasNext()) {
	    t.getElements()[it.next()] = value;
	}
    }

    /**
     * @param nn
     * @param miniBatchSize
     * @param useSharedMemory
     * @return Tensor provider based on neural network
     */
    public static ValuesProvider tensorProvider(NeuralNetwork nn, int miniBatchSize, boolean useSharedMemory) {
	ValuesProvider result = new ValuesProvider(useSharedMemory);

	Map<Layer, Set<int[]>> dims = getLayersDimensions(nn, miniBatchSize);

	// create tensors
	List<Layer> layers = new ArrayList<>(dims.keySet());
	for (int i = 0; i < layers.size(); i++) {
	    Layer l = layers.get(i);
	    for (int[] d : dims.get(l)) {
		result.add(l, d);
	    }
	}

	return result;
    }

    /**
     * @param miniBatchSize
     * @param useSharedMemory
     * @param nns
     * @return Tensor provider based on multiple neural networks - common layers use shared tensors
     */
    public static ValuesProvider tensorProvider(int miniBatchSize, boolean useSharedMemory, NeuralNetwork... nns) {
	ValuesProvider result = new ValuesProvider(useSharedMemory);

	for (NeuralNetwork nn : nns) {
	    Map<Layer, Set<int[]>> dims = getLayersDimensions(nn, miniBatchSize);
	    
	    // create tensors
	    List<Layer> layers = new ArrayList<>(dims.keySet());
	    for (int i = 0; i < layers.size(); i++) {
		Layer l = layers.get(i);
		for (int[] d : dims.get(l)) {
		    if (result.get(l, d) == null) {
			result.add(l, d);
		    }
		}
	    }
	}

	return result;
    }
    
    /**
     * @param sibling
     * @param nn
     * @return Tensor provider based on neural network
     */
    public static ValuesProvider tensorProvider(ValuesProvider sibling, NeuralNetwork nn) {
	Map<Layer, Set<int[]>> dims = getLayersDimensions(nn, batchSize(sibling));
	
	ValuesProvider result = new ValuesProvider(sibling);

	// create tensors
	List<Layer> layers = new ArrayList<>(dims.keySet());
	for (int i = 0; i < layers.size(); i++) {
	    Layer l = layers.get(i);
	    for (int[] d : dims.get(l)) {
		result.add(l, d);
	    }
	}
	
	return result;
    }

    public static void copy(Tensor src, Tensor dest) {
	if (!Arrays.equals(src.getDimensions(), dest.getDimensions())) {
	    throw new IllegalArgumentException("Dimensions don't match");
	}

	TensorIterator srcIt = src.iterator();
	TensorIterator destIt = dest.iterator();
	while (srcIt.hasNext() && destIt.hasNext()) {
	    dest.getElements()[destIt.next()] = src.getElements()[srcIt.next()];
	}
    }

    /**
     * @return mini batch size for TensorProvider
     */
    public static int batchSize(ValuesProvider tp) {
	Tensor t = tp.getTensors().iterator().next();
	return t.getDimensions()[t.getDimensions().length - 1];
    }

    /**
     * @return Tensor for connections. The connections must have a common layer and they must have the same dimensions.
     */
    public static <T extends Tensor> T tensor(Layer targetLayer, Collection<Connections> connections, ValuesProvider tp) {
	return tp.get(targetLayer, getLayerDimensions(targetLayer, connections, batchSize(tp)));
    }

    /**
     * @return Tensor for connections. The connections must have a common layer and they must have the same dimensions.
     */
    public static <T extends Tensor> T tensor(Layer targetLayer, Connections c, ValuesProvider tp) {
	return tp.get(targetLayer, getLayerDimensions(targetLayer, Arrays.asList(new Connections[] {c}), batchSize(tp)));
    }

    /**
     * @param targetLayer
     * @param connections
     * @return
     */
    private static int[] getLayerDimensions(Layer targetLayer, Collection<Connections> connections, int miniBatchSize) {
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
	    result = new int[] {targetLayer.getUnitCount(connections), miniBatchSize };
	} else if (hasSubsampling) {
	    Subsampling2DConnection c = (Subsampling2DConnection) connections.iterator().next();
	    if (c.getOutputLayer() == targetLayer) {
		result = new int[] { c.getFilters(), c.getOutputFeatureMapRows(), c.getOutputFeatureMapColumns(), miniBatchSize };
	    } else if (c.getInputLayer() == targetLayer) {
		result = new int[] { c.getFilters(), c.getInputFeatureMapRows(), c.getInputFeatureMapColumns(), miniBatchSize };
	    }
	} else if (hasConvolutional) {
	    Conv2DConnection c = (Conv2DConnection) connections.iterator().next();
	    if (c.getOutputLayer() == targetLayer) {
		result = new int[] { c.getOutputFilters(), c.getOutputFeatureMapRows(), c.getOutputFeatureMapColumns(), miniBatchSize };
	    } else if (c.getInputLayer() == targetLayer) {
		result = new int[] { c.getInputFilters(), c.getInputFeatureMapRows(), c.getInputFeatureMapColumns(), miniBatchSize };
	    }
	}

	return result;
    }

    private static Map<Layer, Set<int[]>> getLayersDimensions(NeuralNetwork neuralNetwork, int miniBatchSize) {
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
		int[] inputDim = new int[] { fc.getInputUnitCount(), miniBatchSize };
		int[] outputDim = new int[] { fc.getOutputUnitCount(), miniBatchSize };
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
		int[] inputDim = new int[] { cc.getInputFilters(), cc.getInputFeatureMapRows(), cc.getInputFeatureMapColumns(), miniBatchSize };
		int[] outputDim = new int[] { cc.getOutputFilters(), cc.getOutputFeatureMapRows(), cc.getOutputFeatureMapColumns(), miniBatchSize };
		if (!din.stream().anyMatch(i -> Arrays.equals(i, inputDim))) {
		    din.add(inputDim);
		}
		if (!dout.stream().anyMatch(i -> Arrays.equals(i, outputDim))) {
		    dout.add(outputDim);
		}
	    } else if (c instanceof Subsampling2DConnection) {
		Subsampling2DConnection cc = (Subsampling2DConnection) c;
		int[] inputDim = new int[] { cc.getFilters(), cc.getInputFeatureMapRows(), cc.getInputFeatureMapColumns(), miniBatchSize };
		int[] outputDim = new int[] { cc.getFilters(), cc.getOutputFeatureMapRows(), cc.getOutputFeatureMapColumns(), miniBatchSize };
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
