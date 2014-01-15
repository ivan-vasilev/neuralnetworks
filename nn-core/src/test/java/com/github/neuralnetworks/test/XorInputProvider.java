package com.github.neuralnetworks.test;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputProvider;

public class XorInputProvider implements TrainingInputProvider {

    private int count;

    public XorInputProvider(int count) {
	super();
	this.count = count;
    }

    @Override
    public TrainingInputData getNextInput() {
	if (count > 0) {
	    count--;
	    return new XorTrainingInputData();
	}

	return null;
    }

    @Override
    public void reset() {
	// TODO Auto-generated method stub
    }

    @Override
    public int getInputSize() {
	return 1;
    }

    private static class XorTrainingInputData implements TrainingInputData {

	private Matrix input;
	private Matrix target;

	public XorTrainingInputData() {
	    super();
	    input = new Matrix(new float[] { 0, 1, 0, 1, 0, 0, 1, 1 }, 4);
	    target = new Matrix(new float[] { 0, 1, 1, 0 }, 4);
	}

	@Override
	public Matrix getInput() {
	    return input;
	}

	@Override
	public Matrix getTarget() {
	    return target;
	}
    }

    public static class XorOutputError implements OutputError {

	private float networkError;
	private int size;

	@Override
	public void addItem(Matrix networkOutput, Matrix targetOutput) {
	    for (int i = 0; i < targetOutput.getColumns(); i++, size++) {
		networkError += Math.abs(networkOutput.get(0, i) - targetOutput.get(0, i));
	    }
	}

	@Override
	public float getTotalNetworkError() {
	    return size > 0 ? networkError / size : 0;
	}
    }
}
