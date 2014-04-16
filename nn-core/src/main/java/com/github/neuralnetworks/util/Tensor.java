package com.github.neuralnetworks.util;

import java.io.Serializable;
import java.util.Arrays;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

/**
 * N-dimensional tensor. For example 2-dim tensor is a matrix
 */
public class Tensor implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Start offset in case there is different dimentionality
     */
    protected int startOffset;

    /**
     * tensor elements
     */
    protected float[] elements;

    /**
     * global dimension lengths for the tensor (based on the full elements array)
     */
    protected int[] globalDimensions;

    /**
     * local dimension size of the tensor based on the globalDimensions
     */
    protected int[] dimensions;

    /**
     * sub-tensor position limits for each dimension
     */
    protected int[][] globalDimensionsLimit;

    protected int[] dimMultiplicators;

    /**
     * temporary array
     */
    protected int[] dimTmp;

    public Tensor(Tensor parent, int[][] dimensionsLimit) {
	this.globalDimensions = parent.globalDimensions;
	this.elements = parent.elements;
	this.dimMultiplicators = parent.dimMultiplicators;
	this.globalDimensionsLimit = dimensionsLimit;
	this.dimTmp = new int[globalDimensions.length];

	this.dimensions = new int[(int) IntStream.range(0, globalDimensions.length).filter(i -> dimensionsLimit[0][i] != dimensionsLimit[1][i]).count()];
	for (int i = 0, j = 0; i < globalDimensions.length; i++) {
	    if (dimensionsLimit[0][i] != dimensionsLimit[1][i]) {
		dimensions[j++] = dimensionsLimit[1][i] - dimensionsLimit[0][i] + 1;
	    }
	}
    }

    public Tensor(int startOffset, float[] elements, int[] globalDimensions, int[][] globalDimensionsLimit) {
	super();

	if (globalDimensions == null || globalDimensions.length == 0) {
	    throw new IllegalArgumentException("Please provide dimensions");
	}

	this.startOffset = startOffset;
	this.elements = elements;
	this.globalDimensions = globalDimensions;
	this.globalDimensionsLimit = globalDimensionsLimit;
	this.dimTmp = new int[globalDimensions.length];

	this.dimensions = new int[(int) IntStream.range(0, globalDimensions.length).filter(i -> globalDimensionsLimit[0][i] != globalDimensionsLimit[1][i] || globalDimensionsLimit[1][i] - globalDimensionsLimit[0][i] + 1 == globalDimensions[i]).count()];
	for (int i = 0, j = 0; i < globalDimensions.length; i++) {
	    if (globalDimensionsLimit[0][i] != globalDimensionsLimit[1][i] || globalDimensionsLimit[1][i] - globalDimensionsLimit[0][i] + 1 == globalDimensions[i]) {
		dimensions[j++] = globalDimensionsLimit[1][i] - globalDimensionsLimit[0][i] + 1;
	    }
	}

	this.dimMultiplicators = new int[dimensions.length];
	IntStream.range(0, dimMultiplicators.length).forEach(i -> {
	    globalDimensionsLimit[1][i] = dimensions[i] - 1;
	    dimMultiplicators[i] = 1;
	    Arrays.stream(dimensions).skip(i + 1).limit(dimensions.length).forEach(j -> dimMultiplicators[i] *= j);
	});
    }

    public float get(int... d) {
	return elements[getIndex(d)];
    }

    public void set(float value, int... d) {
	elements[getIndex(d)] = value;
    }

    /**
     * @return Number of elements (may be different than elements.length)
     */
    public int getSize() {
	return IntStream.range(0, dimensions.length).map(i -> dimensions[i]).reduce(1, (a, b) -> a * b);
    }
 
    /**
     * @return start index (in the elements array) for this tensor
     */
    public int getStartIndex() {
	Util.fillArray(dimTmp, 0);
	return getIndex(dimTmp);
    }

    /**
     * @return end index (in the elements array) for this tensor
     */
    public int getEndIndex() {
	IntStream.range(0, globalDimensions.length).forEach(i -> dimTmp[i] = globalDimensionsLimit[1][i] - globalDimensionsLimit[0][i]);
	return getIndex(dimTmp);
    }

    /**
     * @param d
     * @return the distance between two neighboring elements in this dimension in the elements array
     */
    public int getDimensionElementsDistance(int d) {
	return dimMultiplicators[d];
    }

    public float[] getElements() {
        return elements;
    }

    public void setElements(float[] elements) {
        this.elements = elements;
    }

    public int[] getDimensions() {
	return dimensions;
    }

    public void setDimensions(int[] dimensions) {
        this.dimensions = dimensions;
    }

    public int getStartOffset() {
        return startOffset;
    }

    public void setStartOffset(int startOffset) {
        this.startOffset = startOffset;
    }

    /**
     * @return iterator over the indexes of the elements array
     */
    public TensorIterator iterator() {
	return new TensorIterator(this);
    }

    /**
     * @return bordered iterator over the indexes of the elements array
     */
    public TensorIterator iterator(int[][] limits) {
	return new TensorIterator(this, limits);
    }

    /**
     * Iterate over all indexes
     * @param action
     */
    public void forEach(IntConsumer action) {
	TensorIterator it = iterator();
	while (it.hasNext()) {
	    action.accept(it.next());
	}
    }

    protected int getIndex(int... d) {
	if (d == null || d.length == 0 || d.length > globalDimensions.length) {
	    throw new IllegalArgumentException("Please provide indices");
	}

	int id = 0;
	for (int i = 0, j = 0; i < globalDimensions.length; i++) {
	    if (globalDimensionsLimit[0][i] != globalDimensionsLimit[1][i] || d.length - j >= globalDimensions.length - i) {
		if (d[j] + globalDimensionsLimit[0][i] > globalDimensionsLimit[1][i]) {
		    throw new IllegalArgumentException("Index out of range: " + i + " -> " + d[j] + "+" + globalDimensionsLimit[0][i] + " to " + globalDimensionsLimit[1][i]);
		}
		
		id += (d[j++] + globalDimensionsLimit[0][i]) * dimMultiplicators[i];
	    } else {
		id += globalDimensionsLimit[0][i] * dimMultiplicators[i];
	    }
	}

	return startOffset + id;
    }

    /**
     * @param d - dimension
     * @return the index of this dimension within the global dimensions
     */
    protected int getDimensionGlobalIndex(int d) {
	int result = d;

	if (IntStream.range(0, globalDimensions.length).filter(i -> globalDimensionsLimit[1][i] - globalDimensionsLimit[0][i] != globalDimensions[i]).findAny().isPresent()) {
	    for (int i = 0, dim = 0; i < globalDimensions.length; i++) {
		if (globalDimensionsLimit[0][i] != globalDimensionsLimit[1][i]) {
		    if (dim == d) {
			result = i;
			break;
		    }

		    dim++;
		}
	    }
	}

	return result;
    }

    /**
     * Iterate over the "real" indexes of the elements array
     */
    public static class TensorIterator implements java.util.Iterator<Integer> {

	private Tensor tensor;
	private int[] currentPosition;
	private int[][] limits;

	public TensorIterator(Tensor tensor) {
	    super();
	    this.tensor = tensor;
	    this.currentPosition = new int[tensor.dimensions.length];
	    this.limits = new int[2][tensor.dimensions.length];
	    IntStream.range(0, tensor.dimensions.length).forEach(i -> limits[1][i] = tensor.dimensions[i] - 1);
	    currentPosition[currentPosition.length - 1] = -1;
	}

	public TensorIterator(Tensor tensor, int[][] limits) {
	    super();
	    this.tensor = tensor;
	    this.currentPosition = new int[tensor.dimensions.length];
	    this.limits = limits;
	    IntStream.range(0, tensor.dimensions.length - 1).forEach(i -> currentPosition[i] = limits[0][i]);
	    currentPosition[currentPosition.length - 1] = limits[0][currentPosition.length - 1] - 1;
	}

	@Override
	public boolean hasNext() {
	    return IntStream.range(0, tensor.dimensions.length).anyMatch(i -> currentPosition[i] < limits[1][i]);
	}

	@Override
	public Integer next() {
	    for (int d = tensor.dimensions.length - 1; d >= 0; d--) {
		if (currentPosition[d] != limits[1][d]) {
		    currentPosition[d]++;
		    break;
		} else {
		    currentPosition[d] = limits[0][d];
		}
	    }

	    return tensor.getIndex(currentPosition);
	}

	public int[] getCurrentPosition() {
	    return currentPosition;
	}
    }
}
