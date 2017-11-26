package com.github.neuralnetworks.training.events;

import com.github.neuralnetworks.calculation.NetworkCalculator;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputProvider;

/**
 * Event, triggered when a single batch finishes training
 */
public class MiniBatchFinishedEvent extends TrainingEvent
{

	private static final long serialVersionUID = -5239379347414855784L;

	private Integer batchCount;
	private TrainingInputData data;
	private ValuesProvider results;
	private TrainingInputProvider inputProvider;

	public MiniBatchFinishedEvent(NetworkCalculator<?> source, TrainingInputData data, ValuesProvider results, Integer batchCount, TrainingInputProvider inputProvider)
	{
		super(source);
		this.data = data;
		this.results = results;
		this.batchCount = batchCount;
		this.inputProvider = inputProvider;
	}

	public TrainingInputData getData()
	{
		return data;
	}

	public ValuesProvider getResults()
	{
		return results;
	}

	public Integer getBatchCount()
	{
		return batchCount;
	}

	public TrainingInputProvider getInputProvider()
	{
		return inputProvider;
	}
}
