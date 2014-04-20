package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;

/**
 * Base Aparapi connection calculator for weighted sum functions (matrix
 * multiplication). If there are multiple inbound connections they are combined
 * in a "single" connection and are calculated simultaneously
 * 
 * !!! IMPORTANT !!! Aparapi only works one-dimensional arrays of primitive data
 * types can only call member methods of the Kernel class itself.
 * 
 * Because of this limitations all the data that is contained in the input
 * connections, weight matrices, input values etc is converted into
 * one-dimensional member arrays of this class
 */
public class AparapiWeightedSum extends AparapiFullyConnected implements ConnectionCalculator {

    private static final long serialVersionUID = 1L;

    public AparapiWeightedSum(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	super(inputConnections, valuesProvider, targetLayer);
    }

    @Override
    public void run() {
	int id = getGlobalId();

	int inputStartPosition = 0, inputRowsStep = 0, inputColumnsStep = 0, weightStartPosition = 0, weightStep = 0, dim = 0;
	float value = 0;

	// each input example
	for (int i = 0; i < miniBatchSize; i++) {
	    // each connection (of the combined connections)
	    value = output[outputStartPosition + id * outputRowStep + i * outputColumnStep];
	    for (int k = 0; k < series; k++) {
		// each element in the row/column
		inputStartPosition = inputStartPositions[k];
		inputRowsStep = inputRowSteps[k];
		inputColumnsStep = inputColumnSteps[k];
		weightStartPosition = weightStartPositions[k] + weightsInitialStep[k] * id;
		weightStep = weightsStep[k];
		dim = weightsSize[k];

		for (int j = 0; j < dim; j++) {
		    value += input[inputStartPosition + j * inputRowsStep + i * inputColumnsStep] * weights[weightStartPosition + j * weightStep];
		}
	    }

	    output[outputStartPosition + id * outputRowStep + i * outputColumnStep] = value;
	}

	after();
    }

    protected void after() {
    }
}
