package com.github.neuralnetworks.training.events;

import com.github.neuralnetworks.calculation.NetworkCalculator;

public class ValidationStartedEvent extends PhaseStartedEvent
{

	private static final long serialVersionUID = -5239379347414855784L;

	public ValidationStartedEvent(NetworkCalculator<?> source)
	{
		super(source);
	}
}
