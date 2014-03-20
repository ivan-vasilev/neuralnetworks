package com.github.neuralnetworks.util;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * N-dimensional tensor. For example 2-dim tensor is a matrix
 */
public class Tensor implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * tensor elements
     */
    protected float[] elements;

    /**
     * dimension lengths for the tensor
     */
    protected int[] dimensions;

    /**
     * sub-tensor start positions for each dimension
     */
    protected int[] dimStart;

    /**
     * sub-tensor end positions for each dimension
     */
    protected int[] dimEnd;

    protected int[] dimMultiplicators;

    /**
     * temporary array
     */
    protected int[] dimTmp;

    public Tensor(int... dimensions) {
	if (dimensions == null || dimensions.length == 0) {
	    throw new IllegalArgumentException("Please provide dimensions");
	}

	this.dimensions = dimensions;
	this.dimStart = new int[dimensions.length];
	this.dimEnd = new int[dimensions.length];
	this.dimTmp = new int[dimensions.length];
	this.elements = new float[IntStream.of(dimensions).reduce(1, (a, b) -> a * b)];
	this.dimMultiplicators = new int[dimensions.length];

	IntStream.range(0, dimensions.length).forEach(i -> {
	    dimEnd[i] = dimensions[i] - 1;
	    dimMultiplicators[i] = 1;
	    Arrays.stream(dimensions).skip(i + 1).limit(dimensions.length).forEach(j -> dimMultiplicators[i] *= j);
	});
    }

    public Tensor(float[] elements, int... dimensions) {
	if (dimensions == null || dimensions.length == 0) {
	    throw new IllegalArgumentException("Please provide dimensions");
	}
	
	this.dimensions = dimensions;
	this.dimStart = new int[dimensions.length];
	this.dimEnd = new int[dimensions.length];
	this.dimTmp = new int[dimensions.length];
	this.elements = elements;
	
	dimMultiplicators = new int[dimensions.length];
	IntStream.range(0, dimMultiplicators.length).forEach(i -> {
	    dimEnd[i] = dimensions[i] - 1;
	    dimMultiplicators[i] = 1;
	    Arrays.stream(dimensions).skip(i + 1).limit(dimensions.length).forEach(j -> dimMultiplicators[i] *= j);
	});
    }
    
    public Tensor(Tensor parent, int[] dimStart, int[] dimEnd) {
	this.dimensions = parent.dimensions;
	this.elements = parent.elements;
	this.dimMultiplicators = parent.dimMultiplicators;
	this.dimStart = dimStart;
	this.dimEnd = dimEnd;
	this.dimTmp = new int[dimensions.length];
    }

    public float get(int... d) {
	return elements[getIndex(d)];
    }

    public void set(float value, int... d) {
	elements[getIndex(d)] = value;
    }

    public float[] getElements() {
        return elements;
    }

    /**
     * @return Number of elements (may be different than
     */
    public int getSize() {
	return IntStream.range(0, dimensions.length).map(i -> getDimensionLength(i)).reduce(1, (a, b) -> a * b);
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
	IntStream.range(0, dimensions.length).forEach(i -> dimTmp[i] = dimEnd[i] - dimStart[i]);
	return getIndex(dimTmp);
    }

    /**
     * @param d
     * @return the length of this dimension
     */
    public int getDimensionLength(int d) {
	return dimEnd[d] - dimStart[d] + 1;
    }

    /**
     * @param d
     * @return the distance between two neighboring elements in this dimension in the elements array
     */
    public int getDimensionElementsDistance(int d) {
	return dimMultiplicators[d];
    }

    public int getDimensionsSize() {
	return dimensions.length;
    }

    protected int getIndex(int... d) {
	if (d == null || d.length == 0 || d.length > dimensions.length) {
	    throw new IllegalArgumentException("Please provide indices");
	}

	int id = IntStream.range(0, d.length).map(i -> {
	    if (d[i] + dimStart[i] > dimEnd[i]) {
		throw new IllegalArgumentException("Index out of range: " + i + " -> " + d[i] + "+" + dimStart[i] + " to " + dimStart[i]);
	    }

	    return (d[i] + dimStart[i]) * dimMultiplicators[i];
	}).sum();

	return id;
    }
}
