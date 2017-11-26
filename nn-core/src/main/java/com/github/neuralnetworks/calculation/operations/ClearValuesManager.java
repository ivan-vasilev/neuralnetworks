package com.github.neuralnetworks.calculation.operations;

import java.util.HashSet;
import java.util.Set;

import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.training.events.PhaseFinishedEvent;
import com.github.neuralnetworks.training.events.PhaseStartedEvent;

/**
 * Manager that maintains the state of cleared/non-cleared arrays
 */
public class ClearValuesManager
{
	private Set<Tensor> cleared;

	private static ClearValuesManager singleton = new ClearValuesManager();

	private ClearValuesManager()
	{
		super();
		this.cleared = new HashSet<>();
	}

	public static ClearValuesManager getInstance()
	{
		return singleton;
	}

	public boolean isCleared(Tensor tensor)
	{
		return cleared.contains(tensor);
	}

	public boolean addToCleared(Tensor tensor)
	{
		return cleared.add(tensor);
	}

	public void reset()
	{
		cleared.clear();
	}

	public static class ClearValuesEventListener implements TrainingEventListener
	{
		private static final long serialVersionUID = 1L;

		@Override
		public void handleEvent(TrainingEvent event)
		{
			if (event instanceof PhaseFinishedEvent || event instanceof PhaseStartedEvent)
			{
				ClearValuesManager.getInstance().reset();
			}
		}
	}
}
