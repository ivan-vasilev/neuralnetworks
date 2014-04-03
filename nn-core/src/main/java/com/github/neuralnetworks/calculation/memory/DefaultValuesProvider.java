package com.github.neuralnetworks.calculation.memory;

import java.util.Arrays;
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

/**
 * Values provider that users separate arrays for each layer and combines the
 * layers with common connections
 */
public class DefaultValuesProvider extends ValuesProvider {

    private static final long serialVersionUID = 1L;

    private NeuralNetwork neuralNetwork;

    public DefaultValuesProvider(NeuralNetwork neuralNetwork) {
	super();
	this.neuralNetwork = neuralNetwork;
    }

    @Override
    protected void createValues() {
	if (values == null) {
	    values = new HashMap<>();
	}

	Map<Layer, Set<int[]>> dims = new HashMap<>();
	neuralNetwork.getLayers().stream().filter(l -> !values.containsKey(l)).forEach(l -> values.put(l, new HashSet<>()));

	for (Connections c : neuralNetwork.getConnections()) {
	    Set<int[]> din = dims.get(c.getInputLayer());
	    if (din == null) {
		dims.put(c.getInputLayer(), din = new HashSet<>());
	    }

	    Set<int[]> dout = dims.get(c.getOutputLayer());
	    if (dout == null) {
		dims.put(c.getInputLayer(), dout = new HashSet<>());
	    }

	    if (c instanceof FullyConnected) {
		FullyConnected fc = (FullyConnected) c;
		int[] inputDim = new int[] { fc.getInputUnitCount(), getMiniBatchSize() };
		int[] outputDim = new int[] { fc.getOutputUnitCount(), getMiniBatchSize() };
		inputDim[0] = fc.getInputUnitCount();
		outputDim[0] = fc.getOutputUnitCount();
		if (!din.stream().anyMatch(i -> Arrays.equals(i, inputDim)) && !values.get(c.getInputLayer()).stream().anyMatch(t -> Arrays.equals(t.getDimensions(), inputDim))) {
		    din.add(inputDim);
		}
		if (!dout.stream().anyMatch(i -> Arrays.equals(i, outputDim)) && !values.get(c.getOutputLayer()).stream().anyMatch(t -> Arrays.equals(t.getDimensions(), outputDim))) {
		    dout.add(outputDim);
		}
	    } else if (c instanceof Conv2DConnection) {
		Conv2DConnection cc = (Conv2DConnection) c;
		int[] inputDim = new int[] { cc.getOutputFilters(), cc.getOutputFeatureMapRows(), cc.getOutputFeatureMapColumns(), getMiniBatchSize() };
		int[] outputDim = new int[] { cc.getInputFilters(), cc.getInputFeatureMapRows(), cc.getInputFeatureMapColumns(), getMiniBatchSize() };
		if (!din.stream().anyMatch(i -> Arrays.equals(i, inputDim)) && !values.get(c.getInputLayer()).stream().anyMatch(t -> Arrays.equals(t.getDimensions(), inputDim))) {
		    din.add(inputDim);
		}
		if (!dout.stream().anyMatch(i -> Arrays.equals(i, outputDim)) && !values.get(c.getOutputLayer()).stream().anyMatch(t -> Arrays.equals(t.getDimensions(), outputDim))) {
		    dout.add(outputDim);
		}
	    } else if (c instanceof Subsampling2DConnection) {
		Subsampling2DConnection cc = (Subsampling2DConnection) c;
		int[] inputDim = new int[] { cc.getFilters(), cc.getOutputFeatureMapRows(), cc.getOutputFeatureMapColumns(), getMiniBatchSize() };
		int[] outputDim = new int[] { cc.getFilters(), cc.getInputFeatureMapRows(), cc.getInputFeatureMapColumns(), getMiniBatchSize() };
		if (!din.stream().anyMatch(i -> Arrays.equals(i, inputDim)) && !values.get(c.getInputLayer()).stream().anyMatch(t -> Arrays.equals(t.getDimensions(), inputDim))) {
		    din.add(inputDim);
		}
		if (!dout.stream().anyMatch(i -> Arrays.equals(i, outputDim)) && !values.get(c.getOutputLayer()).stream().anyMatch(t -> Arrays.equals(t.getDimensions(), outputDim))) {
		    dout.add(outputDim);
		}
	    }
	}

//	Map<Layer, Set<Layer>> combinations = new HashMap<>();
//	neuralNetwork.getLayers().forEach(l -> l.getConnections().forEach(c -> {
//	    Layer o = Util.getOppositeLayer(c, l);
//	}));

	values.entrySet().stream().forEach(e -> e.getValue().forEach(t -> {
	    
	}));
    }
}
