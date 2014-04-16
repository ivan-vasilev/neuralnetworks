package com.github.neuralnetworks.test;

import com.github.neuralnetworks.training.TrainingInputProviderImpl;

/**
 * Simple input provider for testing purposes.
 * Training and target data are two dimensional float arrays
 */
public class SimpleInputProvider extends TrainingInputProviderImpl {

    private static final long serialVersionUID = 1L;

    private float[][] input;
    private float[][] target;

    public SimpleInputProvider(float[][] input, float[][] target) {
	super();

	this.input  = input;
	this.target = target;

//	if (input != null) {
//	    int[] inputDims = Arrays.copyOf(input.getDimensions(), input.getDimensions().length);
//	    inputDims[inputDims.length - 1] = miniBatchSize;
//	    data.setInput(TensorFactory.tensor(inputDims));
//	}
//
//	if (target != null) {
//	    int[] targetDims = Arrays.copyOf(target.getDimensions(), target.getDimensions().length);
//	    targetDims[targetDims.length - 1] = miniBatchSize;
//	    data.setTarget(TensorFactory.tensor(targetDims));
//	}
    }

    @Override
    public int getInputSize() {
	return input.length;
    }

    @Override
    public float[] getNextInput() {
	return input[currentInput % input.length];
    }

    @Override
    public float[] getNextTarget() {
	return target[currentInput % target.length];
    }

//    @Override
//    public TrainingInputData getNextInput() {
//	if (current < count) {
//	    for (int i = 0; i < miniBatchSize; i++, current++) {
//		if (input != null) {
//		    int[] d = input.getDimensions();
//
//		    int[][] lmb = new int[2][d.length];
//		    IntStream.range(0, lmb[1].length).forEach(j -> lmb[1][j] = d[j] - 1);
//		    lmb[0][d.length - 1] = lmb[1][d.length - 1] = i;
//		    TensorIterator mbIt = data.getInput().iterator(lmb);
//
//		    int[][] li = new int[2][d.length];
//		    IntStream.range(0, li[1].length).forEach(j -> li[1][j] = d[j] - 1);
//		    li[0][d.length - 1] = li[1][d.length - 1] = current % d[d.length - 1];
//		    TensorIterator inputIterator = input.iterator(li);
//
//		    while (mbIt.hasNext()) {
//			int mbId = mbIt.next();
//			int inId = inputIterator.next();
//			data.getInput().getElements()[mbId] = input.getElements()[inId];
//		    }
//		}
//
//		if (target != null) {
//		    int[] d = target.getDimensions();
//
//		    int[][] lmb = new int[2][d.length];
//		    IntStream.range(0, lmb[1].length).forEach(j -> lmb[1][j] = d[j] - 1);
//		    lmb[0][d.length - 1] = lmb[1][d.length - 1] = i;
//		    TensorIterator mbIt = data.getTarget().iterator(lmb);
//
//		    int[][] li = new int[2][d.length];
//		    IntStream.range(0, li[1].length).forEach(j -> li[1][j] = d[j] - 1);
//		    li[0][d.length - 1] = li[1][d.length - 1] = current % d[d.length - 1];
//		    TensorIterator targetIterator = target.iterator(li);
//
//		    while (mbIt.hasNext()) {
//			int mbId = mbIt.next();
//			int tId = targetIterator.next();
//			data.getTarget().getElements()[mbId] = target.getElements()[tId];
//		    }
//		}
//	    }
//
//	    return data;
//	}
//
//	return null;
//    }
//
//    private static class SimpleTrainingInputData implements TrainingInputData {
//
//	private static final long serialVersionUID = 1L;
//
//	private Tensor input;
//	private Tensor target;
//
//	public SimpleTrainingInputData(Tensor input, Tensor target) {
//	    super();
//	    this.input = input;
//	    this.target = target;
//	}
//
//	@Override
//	public Tensor getInput() {
//	    return input;
//	}
//
//	public void setInput(Tensor input) {
//	    this.input = input;
//	}
//
//	@Override
//	public Tensor getTarget() {
//	    return target;
//	}
//
//	public void setTarget(Tensor target) {
//	    this.target = target;
//	}
//    }
}
