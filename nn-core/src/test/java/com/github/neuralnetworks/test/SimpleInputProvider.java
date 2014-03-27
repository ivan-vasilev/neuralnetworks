package com.github.neuralnetworks.test;

import java.util.Arrays;
import java.util.stream.IntStream;

import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.util.Matrix;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.Tensor.TensorIterator;

/**
 * Simple input provider for testing purposes.
 * Training and target data are two dimensional float arrays
 */
public class SimpleInputProvider implements TrainingInputProvider {

    private static final long serialVersionUID = 1L;

    private Tensor input;
    private Tensor target;

    private SimpleTrainingInputData data;
    private int count;
    private int miniBatchSize;
    private int current;

    public SimpleInputProvider(Tensor input, Tensor target, int count, int miniBatchSize) {
	super();

	this.input  = input;
	this.target = target;
	this.count = count;
	this.miniBatchSize = miniBatchSize;
	this.data = new SimpleTrainingInputData(null, null);

	if (input != null) {
	    int[] inputDims = Arrays.copyOf(input.getDimensions(), input.getDimensions().length);
	    inputDims[inputDims.length - 1] = miniBatchSize;
	    data.setInput(inputDims.length == 2 ? new Matrix(inputDims[0], inputDims[1]) : new Tensor(inputDims));
	}

	if (target != null) {
	    int[] targetDims = Arrays.copyOf(target.getDimensions(), target.getDimensions().length);
	    targetDims[targetDims.length - 1] = miniBatchSize;
	    data.setTarget(targetDims.length == 2 ? new Matrix(targetDims[0], targetDims[1]) : new Tensor(targetDims));
	}
    }

    @Override
    public int getInputSize() {
	return count;
    }

    @Override
    public void reset() {
	current = 0;
    }

    @Override
    public TrainingInputData getNextInput() {
	if (current < count) {
	    for (int i = 0; i < miniBatchSize; i++, current++) {
		if (input != null) {
		    int[] d = input.getDimensions();

		    int[][] lmb = new int[2][d.length];
		    IntStream.range(0, lmb[1].length).forEach(j -> lmb[1][j] = d[j] - 1);
		    lmb[0][lmb.length - 1] = lmb[1][lmb.length - 1] = i;
		    TensorIterator mbIt = data.getInput().iterator(lmb);

		    int[][] li = new int[2][d.length];
		    IntStream.range(0, li[1].length).forEach(j -> li[1][j] = d[j] - 1);
		    li[0][li.length - 1] = li[1][li.length - 1] = current % d[d.length - 1];
		    TensorIterator inputIterator = input.iterator(li);

		    while (mbIt.hasNext()) {
			int mbId = mbIt.next();
			int inId = inputIterator.next();
			data.getInput().getElements()[mbId] = input.getElements()[inId];
		    }
		}

		if (target != null) {
		    int[] d = target.getDimensions();

		    int[][] lmb = new int[2][d.length];
		    IntStream.range(0, lmb[1].length).forEach(j -> lmb[1][j] = d[j] - 1);
		    lmb[0][lmb.length - 1] = lmb[1][lmb.length - 1] = i;
		    TensorIterator mbIt = data.getTarget().iterator(lmb);

		    int[][] li = new int[2][d.length];
		    IntStream.range(0, li[1].length).forEach(j -> li[1][j] = d[j] - 1);
		    li[0][li.length - 1] = li[1][li.length - 1] = current % d[d.length - 1];
		    TensorIterator targetIterator = target.iterator(li);

		    while (mbIt.hasNext()) {
			int mbId = mbIt.next();
			int tId = targetIterator.next();
			data.getTarget().getElements()[mbId] = target.getElements()[tId];
		    }
		}
	    }

	    return data;
	}

	return null;
    }

    private static class SimpleTrainingInputData implements TrainingInputData {

	private static final long serialVersionUID = 1L;

	private Tensor input;
	private Tensor target;

	public SimpleTrainingInputData(Tensor input, Tensor target) {
	    super();
	    this.input = input;
	    this.target = target;
	}

	@Override
	public Tensor getInput() {
	    return input;
	}

	public void setInput(Tensor input) {
	    this.input = input;
	}

	@Override
	public Tensor getTarget() {
	    return target;
	}

	public void setTarget(Tensor target) {
	    this.target = target;
	}
    }
}
