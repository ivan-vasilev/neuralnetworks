package com.github.neuralnetworks.calculation.operations;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiConv2D;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiFullyConnected;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiSubsampling2D;
import com.github.neuralnetworks.tensor.ValuesProvider;

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
public abstract class ConnectionCalculatorImpl implements ConnectionCalculator, ConnectionCalculatorTensorFunctions
{

	private static final long serialVersionUID = -5405654469496055017L;

	protected transient List<ConnectionCalculator> inputFunctions;
	protected transient Map<Connections, ConnectionCalculator> connectionCalculators;

	/**
	 * Activation functions that are executed before the transfer function
	 */
	protected List<TensorFunction> inputModifierFunctions;

	/**
	 * Activation functions that are called after the transfer function
	 */
	protected List<TensorFunction> activationFunctions;

	public ConnectionCalculatorImpl()
	{
		super();
		init();
		this.inputModifierFunctions = new ArrayList<>();
		this.activationFunctions = new ArrayList<>();
	}

	private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException
	{
		stream.defaultReadObject();
		init();
	}

	protected void init()
	{
		this.inputFunctions = new ArrayList<>();
		this.connectionCalculators = new HashMap<>();
	}

	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		if (connections.size() > 0)
		{
			calculateInputModifierFunctions(connections, targetLayer, valuesProvider);

			List<Connections> connection = new ArrayList<>();
			for (Connections c : connections) {
				ConnectionCalculator cc = connectionCalculators.get(c);
				connection.clear();
				connection.add(c);
				if 	(cc == null ||
						(cc != null && cc instanceof AparapiFullyConnected && !((AparapiFullyConnected) cc).accept(connection, valuesProvider, targetLayer)) ||
						(cc != null && cc instanceof AparapiConv2D && !((AparapiConv2D) cc).accept(connection, valuesProvider)) || 
						(cc != null && cc instanceof AparapiSubsampling2D && !((AparapiSubsampling2D) cc).accept((Subsampling2DConnection) c, valuesProvider)))
				{
					connectionCalculators.put(c, cc = createInputFunction(connection, valuesProvider, targetLayer));
					inputFunctions.add(cc);
				}

				cc.calculate(connection, valuesProvider, targetLayer);
			}

			calculateActivationFunctions(connections, targetLayer, valuesProvider);
		}
	}

	@Override
	public List<ConnectionCalculator> getInputFunctions()
	{
		return inputFunctions;
	}

	@Override
	public void setInputFunctions(List<ConnectionCalculator> inputFunctions)
	{
		this.inputFunctions = inputFunctions;
	}

	@Override
	public List<TensorFunction> getInputModifierFunctions()
	{
		return inputModifierFunctions;
	}

	@Override
	public void setInputModifierFunctions(List<TensorFunction> inputModifierFunctions)
	{
		this.inputModifierFunctions = inputModifierFunctions;
	}

	@Override
	public List<TensorFunction> getActivationFunctions()
	{
		return activationFunctions;
	}

	@Override
	public void setActivationFunctions(List<TensorFunction> activationFunctions)
	{
		this.activationFunctions = activationFunctions;
	}

	protected abstract ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer);
}
