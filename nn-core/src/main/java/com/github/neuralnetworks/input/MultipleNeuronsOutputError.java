package com.github.neuralnetworks.input;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.OutputError;

public class MultipleNeuronsOutputError implements OutputError {

    private static final long serialVersionUID = 1L;

    private List<OutputTargetTuple> tuples;
    private Map<Integer, Integer> outputToTarget;
    private int nullCount;
    private int dim;

    public MultipleNeuronsOutputError() {
	super();
	reset();
    }

    @Override
    public void addItem(Matrix networkOutput, Matrix targetOutput) {
	if (dim == -1) {
	    dim = targetOutput.getRows();
	}

	if (networkOutput.getRows() != dim || targetOutput.getRows() != dim) {
	    throw new IllegalArgumentException("Dimensions don't match");
	}

	for (int i = 0; i < targetOutput.getColumns(); i++) {
	    boolean hasDifferentValues = false;
	    for (int j = 0; j < networkOutput.getRows(); j++) {
		if (networkOutput.get(j, i) != networkOutput.get(0, i)) {
		    hasDifferentValues = true;
		    break;
		}
	    }

	    if (hasDifferentValues) {
		int targetPos = 0;
		for (int j = 0; j < targetOutput.getRows(); j++) {
		    if (targetOutput.get(j, i) == 1) {
			targetPos = j;
			break;
		    }
		}

		int outputPos = 0;
		float max = networkOutput.get(0, i);
		for (int j = 0; j < networkOutput.getRows(); j++) {
		    if (networkOutput.get(j, i) > max) {
			max = networkOutput.get(j, i);
			outputPos = j;
		    }
		}

		tuples.add(new OutputTargetTuple(outputPos, targetPos));
	    } else {
		nullCount++;
	    }
	}
    }

    @Override
    public float getTotalNetworkError() {
	return getTotalInputSize() > 0 ? ((float) getTotalErrorSamples()) / getTotalInputSize() : 0;
    }

    @Override
    public int getTotalErrorSamples() {
	if (outputToTarget == null) {
	    outputToTarget = outputToTarget();
	}

	int errorSamples = 0;
	for (OutputTargetTuple t : tuples) {
	    if (!outputToTarget.get(t.outputPos).equals(t.targetPos)) {
		errorSamples++;
	    }
	}

	return nullCount + errorSamples;
    }

    @Override
    public int getTotalInputSize() {
	return tuples.size() + nullCount;
    }

    private Map<Integer, Integer> outputToTarget() {
	Map<Integer, Integer> result = new HashMap<>();
	Map<Integer, int[]> targetToOutput = new HashMap<>();
	for (OutputTargetTuple t : tuples) {
	    if (!targetToOutput.containsKey(t.targetPos)) {
		targetToOutput.put(t.targetPos, new int[dim]);
	    }

	    targetToOutput.get(t.targetPos)[t.outputPos]++;
	}

	for (int i = 0; i < dim; i++) {
	    int[] d = targetToOutput.get(i);
	    if (d != null) {
		int max = 0;
		for (int j = 0; j < dim; j++) {
		    if (d[j] > d[max] && !result.values().contains(j)) {
			max = j;
		    }
		}

		result.put(i, max);
	    }
	}

	return result;
    }

    @Override
    public void reset() {
	this.tuples = new ArrayList<>();
	this.dim = -1;
	this.nullCount = 0;
	this.outputToTarget = null;
    }

    private static class OutputTargetTuple {

	public OutputTargetTuple(Integer outputPos, Integer targetPos) {
	    this.outputPos = outputPos;
	    this.targetPos = targetPos;
	}

	public Integer outputPos;
	public Integer targetPos;
    }
}
