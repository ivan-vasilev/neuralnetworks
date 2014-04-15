package com.github.neuralnetworks.training;

import java.io.Serializable;
import java.util.List;
import java.util.stream.IntStream;

import com.github.neuralnetworks.calculation.neuronfunctions.TensorFunction;
import com.github.neuralnetworks.util.Tensor.TensorIterator;

/**
 * Input provider for training data
 */
public interface TrainingInputProvider extends Serializable {

    public int getInputSize();
    public void reset();
    float[] getNextInput();
    float[] getNextTarget();
    List<TensorFunction> getInputModifiers();
    void after(TrainingInputData ti);

    public default void populateNext(TrainingInputData ti) {
	// input
	if (ti.getInput() != null) {
	    int[] inputDims = ti.getInput().getDimensions();
	    int[][] limits = new int[2][inputDims.length];
	    IntStream.range(0, inputDims.length - 1).forEach(i -> limits[1][i] = inputDims[i]);
	    IntStream.range(0, inputDims[inputDims.length - 1]).forEach(i -> {
		limits[0][inputDims.length - 1] = limits[1][inputDims.length - 1] = i;
		TensorIterator it = ti.getInput().iterator(limits);
		float[] inputEl = getNextInput();
		IntStream.range(0, inputEl.length).forEach(j -> ti.getInput().getElements()[it.next()] = inputEl[j]);
	    });

	    if (getInputModifiers() != null) {
		getInputModifiers().forEach(im -> im.value(ti.getInput()));
	    }
	}

	// target
	if (ti.getTarget() != null) {
	    int[] targetDims = ti.getTarget().getDimensions();
	    int[][] limits = new int[2][targetDims.length];
	    IntStream.range(0, targetDims.length - 1).forEach(i -> limits[1][i] = targetDims[i]);
	    IntStream.range(0, targetDims[targetDims.length - 1]).forEach(i -> {
		limits[0][targetDims.length - 1] = limits[1][targetDims.length - 1] = i;
		TensorIterator it = ti.getTarget().iterator(limits);
		float[] targetEl = getNextTarget();
		IntStream.range(0, targetEl.length).forEach(j -> ti.getTarget().getElements()[it.next()] = targetEl[j]);
	    });
	}

	after(ti);
    }
}
