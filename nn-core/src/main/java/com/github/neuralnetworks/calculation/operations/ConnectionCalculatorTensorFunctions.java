package com.github.neuralnetworks.calculation.operations;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.events.PropagationEvent;
import com.github.neuralnetworks.events.PropagationEventListener;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.Util;

/**
 * Convenience interface for activation functions
 */

public interface ConnectionCalculatorTensorFunctions extends PropagationEventListener
{
	public List<ConnectionCalculator> getInputFunctions();

	public void setInputFunctions(List<ConnectionCalculator> inputFunctions);

	public default void calculateInputModifierFunctions(List<Connections> connections, Layer targetLayer, ValuesProvider valuesProvider)
	{
		if (getInputModifierFunctions() != null && getInputModifierFunctions().size() > 0)
		{
			getInputModifierFunctions().forEach(f -> connections.stream().filter(c -> !Util.isBias(c.getInputLayer()))
					.forEach(c -> f.value(TensorFactory.tensor(Util.getOppositeLayer(c, targetLayer), c, valuesProvider))));
		}
	}

	public default void calculateActivationFunctions(List<Connections> connections, Layer targetLayer, ValuesProvider valuesProvider)
	{
		if (getActivationFunctions() != null && getActivationFunctions().size() > 0)
		{
			getActivationFunctions().forEach(f -> f.value(TensorFactory.tensor(targetLayer, connections, valuesProvider)));
		}
	}

	@Override
	public default void handleEvent(PropagationEvent event)
	{
		if (getInputModifierFunctions() != null)
		{
			getInputModifierFunctions().stream().filter(f -> f instanceof PropagationEventListener).forEach(f -> ((PropagationEventListener) f).handleEvent(event));
		}

		if (getActivationFunctions() != null)
		{
			getActivationFunctions().stream().filter(f -> f instanceof PropagationEventListener).forEach(f -> ((PropagationEventListener) f).handleEvent(event));
		}
	}

	public default void addInputModifierFunction(TensorFunction function)
	{
		if (getInputModifierFunctions() == null)
		{
			setInputModifierFunctions(new UniqueList<>());
		}

		getInputModifierFunctions().add(function);
	}

	public default void removeInputModifier(TensorFunction function)
	{
		if (getInputModifierFunctions() != null)
		{
			getInputModifierFunctions().remove(function);
		}
	}

	public default void addActivationFunction(TensorFunction activationFunction)
	{
		if (getActivationFunctions() == null)
		{
			setActivationFunctions(new UniqueList<>());
		}

		getActivationFunctions().add(activationFunction);
	}

	public default void removeActivationFunction(TensorFunction activationFunction)
	{
		if (getActivationFunctions() != null)
		{
			getActivationFunctions().remove(activationFunction);
		}
	}

	public List<TensorFunction> getInputModifierFunctions();

	public void setInputModifierFunctions(List<TensorFunction> inputModifierFunctions);

	public List<TensorFunction> getActivationFunctions();

	public void setActivationFunctions(List<TensorFunction> activationFunctions);
}
