package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.tensor.ValuesProvider;

public interface BackPropagationConnectionCalculator extends ConnectionCalculator
{
	public ValuesProvider getActivations();
	public void setActivations(ValuesProvider activations);
}
