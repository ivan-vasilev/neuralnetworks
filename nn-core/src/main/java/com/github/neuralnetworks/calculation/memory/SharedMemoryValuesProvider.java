package com.github.neuralnetworks.calculation.memory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.TensorFactory;

/**
 * Values provider that users separate arrays for each layer and combines the
 * layers with common connections
 */
public class SharedMemoryValuesProvider extends ValuesProvider {

    private static final long serialVersionUID = 1L;

    private NeuralNetwork neuralNetwork;

    public SharedMemoryValuesProvider(NeuralNetwork neuralNetwork) {
	super();
	this.neuralNetwork = neuralNetwork;
    }

    @Override
    protected void createValues(Map<Layer, Set<Tensor>> values) {
	Map<Layer, Set<int[]>> dims = getLayerDimensions(neuralNetwork);

	// remove existing
	values.entrySet().forEach(e -> {
	    dims.get(e.getKey()).removeIf(d -> e.getValue().stream().anyMatch(t -> Arrays.equals(t.getDimensions(), d)));
	    if (dims.get(e.getKey()).size() == 0) {
		dims.remove(e.getKey());
	    }
	});

	// common elements array
	float[] elements = new float[dims.values().stream().map(s -> Arrays.stream(s.iterator().next()).reduce(1, (a, b) -> a * b)).reduce(0, (a, b) -> a + b)];

	// create tensors
	List<Layer> layers = new ArrayList<>(dims.keySet());
	for (int i = 0, offset = 0; i < layers.size(); i++) {
	    Layer l = layers.get(i);
	    for (int[] d : dims.get(l)) {
		if (!values.containsKey(l)) {
		    values.put(l, new HashSet<>());
		}
		values.get(l).add(TensorFactory.tensor(elements, offset, d));
	    }

	    offset += Arrays.stream(dims.get(l).iterator().next()).reduce(1, (a, b) -> a * b);
	}
    }
}
