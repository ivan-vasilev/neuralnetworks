package com.github.neuralnetworks.training.events;

import com.github.neuralnetworks.training.Trainer;

public class TestingFinishedEvent extends PhaseFinishedEvent
{

	private static final long serialVersionUID = -5239379347414855784L;

	public TestingFinishedEvent(Trainer<?> source)
	{
		super(source);
	}
}
