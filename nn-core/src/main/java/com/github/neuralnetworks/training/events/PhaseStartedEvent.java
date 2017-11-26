package com.github.neuralnetworks.training.events;

import com.github.neuralnetworks.calculation.NetworkCalculator;
import com.github.neuralnetworks.events.TrainingEvent;

public class PhaseStartedEvent extends TrainingEvent
{

	private static final long serialVersionUID = -5239379347414855784L;

	public PhaseStartedEvent(NetworkCalculator<?> source)
	{
		super(source);
	}
}
