package com.github.neuralnetworks.architecture;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.github.neuralnetworks.util.Matrix;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.TensorFactory;

/**
 * Factory for connections. In order to use shared weights it cannot be static.
 */
public class ConnectionFactory implements Serializable {

    private static final long serialVersionUID = 1L;

    private List<Connections> connections;
    private float[] sharedWeights;

    public ConnectionFactory() {
	super();
    }

    public ConnectionFactory(boolean useSharedMemory) {
	super();
	if (useSharedMemory) {
	    this.connections = new ArrayList<>();
	    this.sharedWeights = new float[0];
	}

    }

    public FullyConnected fullyConnected(Layer inputLayer, Layer outputLayer, int inputUnitCount, int outputUnitCount) {
	Matrix weights = null;
	if (useSharedWeights()) {
	    int l = sharedWeights.length;
	    sharedWeights = Arrays.copyOf(sharedWeights, l + inputUnitCount * outputUnitCount);
	    updateSharedWeights();
	    weights = TensorFactory.tensor(sharedWeights, l, outputUnitCount, inputUnitCount);
	} else {
	    weights = TensorFactory.tensor(outputUnitCount, inputUnitCount);
	}

	return fullyConnected(inputLayer, outputLayer, weights);
    }

    public FullyConnected fullyConnected(Layer inputLayer, Layer outputLayer, Matrix weights) {
	FullyConnected result = new FullyConnected(inputLayer, outputLayer, weights);
	if (useSharedWeights()) {
	    connections.add(result);
	}

	return result;
    }

    public Conv2DConnection conv2d(Layer inputLayer, Layer outputLayer, int inputFeatureMapColumns, int inputFeatureMapRows, int inputFilters, int kernelRows, int kernelColumns, int outputFilters, int stride) {
	Tensor weights = null;
	if (useSharedWeights()) {
	    int l = sharedWeights.length;
	    sharedWeights = Arrays.copyOf(sharedWeights, l + outputFilters * inputFilters * kernelRows * kernelColumns);
	    updateSharedWeights();
	    weights = TensorFactory.tensor(sharedWeights, l, outputFilters, inputFilters, kernelRows, kernelColumns);
	} else {
	    weights = TensorFactory.tensor(outputFilters, inputFilters, kernelRows, kernelColumns);
	}

	return conv2d(inputLayer, outputLayer, inputFeatureMapColumns, inputFeatureMapRows, weights, stride);
    }

    public Conv2DConnection conv2d(Layer inputLayer, Layer outputLayer, int inputFeatureMapColumns, int inputFeatureMapRows, Tensor weights, int stride) {
	Conv2DConnection result = new Conv2DConnection(inputLayer, outputLayer, inputFeatureMapColumns, inputFeatureMapRows, weights, stride);
	if (useSharedWeights()) {
	    connections.add(result);
	}

	return result;
    }

    public Subsampling2DConnection subsampling2D(Layer inputLayer, Layer outputLayer, int inputFeatureMapColumns, int inputFeatureMapRows, int subsamplingRegionRows, int subsamplingRegionCols, int filters) {
	return new Subsampling2DConnection(inputLayer, outputLayer, inputFeatureMapColumns, inputFeatureMapRows, subsamplingRegionRows, subsamplingRegionCols, filters);
    }

    private boolean useSharedWeights() {
	return sharedWeights != null;
    }

    private void updateSharedWeights() {
	for (Connections c : connections) {
	    Tensor weights = null;
	    if (c instanceof FullyConnected) {
		weights = ((FullyConnected) c).getConnectionGraph();
	    } else if (c instanceof Conv2DConnection) {
		weights = ((Conv2DConnection) c).getWeights();
	    }

	    if (weights != null) {
		weights.setElements(sharedWeights);
	    }
	}
    }
}
