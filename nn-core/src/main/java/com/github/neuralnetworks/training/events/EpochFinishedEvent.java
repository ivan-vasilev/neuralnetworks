package com.github.neuralnetworks.training.events;

import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainingInputData;

/**
 * Event, triggered when a single epoch finishes training
 */
public class EpochFinishedEvent extends TrainingEvent
{

	private static final long serialVersionUID = -5239379347414855784L;

	private Integer batchCount;
	private Integer epoch;
	private TrainingInputData data;
	private ValuesProvider results;

	public EpochFinishedEvent(Trainer<?> source, TrainingInputData data, ValuesProvider results, Integer batchCount, Integer epoch)
	{
		super(source);
		this.data = data;
		this.results = results;
		this.batchCount = batchCount;
		this.epoch = epoch;
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

	public Integer getEpoch()
	{
		return epoch;
	}
}
