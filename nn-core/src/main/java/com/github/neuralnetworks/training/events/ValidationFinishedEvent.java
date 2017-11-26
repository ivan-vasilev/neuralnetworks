package com.github.neuralnetworks.training.events;

import com.github.neuralnetworks.calculation.NetworkCalculator;

public class ValidationFinishedEvent extends PhaseFinishedEvent
{

	private static final long serialVersionUID = -5239379347414855784L;

	public ValidationFinishedEvent(NetworkCalculator<?> source)
	{
		super(source);
	}
}
