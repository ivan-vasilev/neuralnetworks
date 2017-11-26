package com.github.neuralnetworks.training.events;

import com.github.neuralnetworks.training.Trainer;

public class TestingStartedEvent extends PhaseStartedEvent
{
	private static final long serialVersionUID = -5239379347414855784L;

	public TestingStartedEvent(Trainer<?> source)
	{
		super(source);
	}
}
