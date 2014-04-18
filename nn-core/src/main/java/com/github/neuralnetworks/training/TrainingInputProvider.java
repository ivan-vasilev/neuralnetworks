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

    void afterBatch(TrainingInputData ti);

    void beforeBatch(TrainingInputData ti);
    
    void afterSample();
    
    void beforeSample();

    public default void populateNext(TrainingInputData ti) {
	beforeBatch(ti);

	// batch size
	int batchSize = 0;
	if (ti.getInput() != null && ti.getTarget() != null && ti.getInput().getDimensions()[ti.getInput().getDimensions().length - 1] != ti.getInput().getDimensions()[ti.getInput().getDimensions().length - 1]) {
	    throw new IllegalArgumentException("Input and target batch size don't match");
	}

	if (ti.getInput() != null) {
	    batchSize = ti.getInput().getDimensions()[ti.getInput().getDimensions().length - 1];
	} else if (ti.getTarget() != null) {
	    batchSize = ti.getTarget().getDimensions()[ti.getTarget().getDimensions().length - 1];
	}

	int[] inputDims = null;
	int[][] inputLimits = null;
	int[] targetDims = null;
	int[][] targetLimits = null;

	if (ti.getInput() != null) {
	    inputDims = ti.getInput().getDimensions();
	    inputLimits = new int[2][inputDims.length];
	    for (int i = 0; i < inputDims.length - 1; i++) {
		inputLimits[1][i] = inputDims[i] - 1;
	    }
	}

	if (ti.getTarget() != null) {
	    targetDims = ti.getTarget().getDimensions();
	    targetLimits = new int[2][targetDims.length];
	    for (int i = 0; i < targetDims.length - 1; i++) {
		targetLimits[1][i] = targetDims[i] - 1;
	    }
	}

	// data population
	for (int i = 0; i < batchSize; i++) {
	    beforeSample();

	    if (ti.getInput() != null) {
		inputLimits[0][inputDims.length - 1] = inputLimits[1][inputDims.length - 1] = i;
		TensorIterator inputIt = ti.getInput().iterator(inputLimits);
		float[] inputEl = getNextInput();
		IntStream.range(0, inputEl.length).forEach(j -> ti.getInput().getElements()[inputIt.next()] = inputEl[j]);
	    }

	    if (ti.getTarget() != null) {
		targetLimits[0][targetDims.length - 1] = targetLimits[1][targetDims.length - 1] = i;
		TensorIterator targetIt = ti.getTarget().iterator(targetLimits);
		float[] targetEl = getNextTarget();
		IntStream.range(0, targetEl.length).forEach(j -> ti.getTarget().getElements()[targetIt.next()] = targetEl[j]);
	    }

	    afterSample();
	}

	if (ti.getInput() != null && getInputModifiers() != null) {
	    getInputModifiers().forEach(im -> im.value(ti.getInput()));
	}

	afterBatch(ti);
    }
}
