package com.github.neuralnetworks.builder.activation;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorTensorFunctions;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;

/**
 * @author tmey
 */
public class LayerUtil
{

	public static void changeActivationAndTransferFunction(LayerCalculator layerCalculator, Layer layer, TransferFunctionType transferFunctionType, ActivationType activationType)
	{

		if (!(layerCalculator instanceof LayerCalculatorImpl))
		{
			throw new IllegalArgumentException("LayerCalculator type not supported (" + layerCalculator.getClass() + ")");
		}

		LayerCalculatorImpl layerCalculatorImpl = (LayerCalculatorImpl) layerCalculator;

		ConnectionCalculator connectionCalculator;

		// create ConnectionCalculator with specific transfer function

		switch (transferFunctionType) {
		case Max:
			connectionCalculator = OperationsFactory.maxout();
			break;
		case Max_Polling2D:
			connectionCalculator = OperationsFactory.maxPooling2D();
			break;
		case Average_Pooling2D:
			connectionCalculator = OperationsFactory.averagePooling2D();
			break;
		case WeightedSum:
			connectionCalculator = OperationsFactory.weightedSum();
			break;
		case Conv2D:
			connectionCalculator = OperationsFactory.conv2D();
			break;
		default:
			throw new IllegalArgumentException("unknown transfer function type: " + transferFunctionType);
		}

		// add activation function

		switch (activationType) {
		case ReLU:
			((ConnectionCalculatorTensorFunctions) connectionCalculator).addActivationFunction(OperationsFactory.reLUFunction());
			break;
		case Sigmoid:
			((ConnectionCalculatorTensorFunctions) connectionCalculator).addActivationFunction(OperationsFactory.sigmoidFunction());
			break;
		case SoftMax:
			((ConnectionCalculatorTensorFunctions) connectionCalculator).addActivationFunction(OperationsFactory.softmaxFunction());
			break;
		case Nothing:
			// default
			break;
		default:
			throw new IllegalArgumentException("unknown activation function type: " + activationType);
		}

		layerCalculatorImpl.addConnectionCalculator(layer, connectionCalculator);

	}
}
