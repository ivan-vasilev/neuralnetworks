package com.github.neuralnetworks.calculation.operations.opencl;

import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;

/**
 * Simple wrapper over OpenCLKernelsExecutor that allows serialization
 */
public class OpenCLManagementListener implements TrainingEventListener
{
	private static final long serialVersionUID = 1L;

	@Override
	public void handleEvent(TrainingEvent event)
	{
		OpenCLKernelsExecutor.getInstance().handleEvent(event);
	}
}
