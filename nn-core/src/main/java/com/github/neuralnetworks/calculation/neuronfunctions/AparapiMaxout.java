package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.util.TensorFactory;

/**
 * Maxout activation
 */
public class AparapiMaxout extends ConnectionCalculatorFullyConnected {

    private static final long serialVersionUID = -6602713983386107132L;

    @Override
    protected ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	return new AparapiMaxoutFunction(inputConnections, valuesProvider, targetLayer);
    }

    public static class AparapiMaxoutFunction extends AparapiFullyConnected {

	private static final long serialVersionUID = 2572354641295173835L;

	private final int[] winnersStartPositions;
	private final int[] maxoutWinners;

	public AparapiMaxoutFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	    super(inputConnections, valuesProvider, targetLayer);
	    MaxoutWinners.getInstance().setBatchSize(TensorFactory.batchSize(valuesProvider));
	    winnersStartPositions = MaxoutWinners.getInstance().getStartPositions(inputConnections);
	    maxoutWinners = MaxoutWinners.getInstance().getWinners();
	}

	@Override
	public void run() {
	    int id = getGlobalId();

	    int inputStartPosition = 0, inputRowsStep = 0, inputColumnsStep = 0, weightStartPosition = 0, weightStep = 0, dim = 0, maxIndex = 0;
	    float max = 0, current = 0;;

	    // each input example
	    for (int i = 0; i < miniBatchSize; i++) {
		// each connection (of the combined connections)
		for (int k = 0; k < series; k++) {
		    // each element in the row/column
		    inputStartPosition = inputStartPositions[k];
		    inputRowsStep = inputRowSteps[k];
		    inputColumnsStep = inputColumnSteps[k];
		    weightStartPosition = weightStartPositions[k] + weightsInitialStep[k] * id;
		    weightStep = weightsStep[k];
		    dim = weightsSize[k];

		    maxIndex = 0;
		    current = max = input[inputStartPosition + i * inputColumnsStep] * weights[weightStartPosition];
		    for (int j = 1; j < dim; j++) {
			current = input[inputStartPosition + j * inputRowsStep + i * inputColumnsStep] * weights[weightStartPosition + j * weightStep];
			if (current > max) {
			    max = current;
			    maxIndex = j;
			}
		    }

		    maxoutWinners[winnersStartPositions[k] + i] = maxIndex;
		}

		output[outputStartPosition + id * outputRowStep + i * outputColumnStep] += max;
	    }
	}
    }
}
