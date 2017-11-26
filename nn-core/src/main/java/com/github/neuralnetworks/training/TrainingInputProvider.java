package com.github.neuralnetworks.training;

import java.io.Serializable;
import java.util.List;

import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Tensor;

/**
 * Input provider for training data
 */
public interface TrainingInputProvider extends Serializable
{

	public int getInputSize();

	public int getInputDimensions();

	public int getTargetDimensions();

	public void reset();

	void getNextInput(Tensor input);

	void getNextTarget(Tensor target);

	List<TensorFunction> getInputModifiers();

	void beforeBatch(TrainingInputData ti);

	void afterBatch(TrainingInputData ti);

	public default void populateNext(TrainingInputData ti)
	{
		beforeBatch(ti);

		// batch size
		if (ti.getInput() != null && ti.getTarget() != null && ti.getInput().getDimensions()[0] != ti.getTarget().getDimensions()[0])
		{
			throw new IllegalArgumentException("Input and target batch size don't match");
		}

		if (ti.getInput() != null)
		{
			getNextInput(ti.getInput());
		}

		if (ti.getTarget() != null)
		{
			getNextTarget(ti.getTarget());
		}

		if (ti.getInput() != null && getInputModifiers() != null)
		{
			getInputModifiers().forEach(im -> im.value(ti.getInput()));
		}

		afterBatch(ti);
	}
}
