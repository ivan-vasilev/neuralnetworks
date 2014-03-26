package com.github.neuralnetworks.util;

import java.util.stream.IntStream;

/**
 * Simple matrix representation with one-dimensional array. This is required,
 * because Aparapi supports only one-dim arrays (otherwise the execution is
 * transferred to the cpu)
 */
public class Matrix extends Tensor {

    private static final long serialVersionUID = 1L;

    public Matrix() {
	super();
    }

    /**
     * simplified constructor
     */
    public Matrix(float[][] elements) {
	super(elements[0].length, elements.length);
	IntStream.range(0, elements.length).forEach(i -> IntStream.range(0, elements[i].length).forEach(j -> {
	    set(elements[i][j], j, i);
	}));
    }

    public Matrix(float[] elements, int columns) {
	super(elements, elements.length / columns, columns);
    }

    public Matrix(int rows, int columns) {
	super(rows, columns);
    }

    public Matrix(Matrix copy) {
	super(copy.getRows(), copy.getColumns());
    }

    public Matrix(Tensor parent, int[][] dimensionsLimit) {
	super(parent, dimensionsLimit);
    }

    public int getColumns() {
	return getDimensions()[1];
    }

    public int getColumnElementsDistance() {
	return getDimensionElementsDistance(1);
    }

    public int getRows() {
	return getDimensions()[0];
    }

    public int getRowElementsDistance() {
	return getDimensionElementsDistance(0);
    }
}
