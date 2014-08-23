package com.github.neuralnetworks.tensor;


/**
 * Simple matrix representation with one-dimensional array. This is required,
 * because Aparapi supports only one-dim arrays (otherwise the execution is
 * transferred to the cpu)
 */
public class Matrix extends Tensor {

    private static final long serialVersionUID = 1L;

    public Matrix(Tensor parent, int[][] dimensionsLimit) {
	super(parent, dimensionsLimit);
    }

    public Matrix(int startOffset, float[] elements, int[] globalDimensions, int[][] globalDimensionsLimit) {
	super(startOffset, elements, globalDimensions, globalDimensionsLimit);
    }

    public int getColumns() {
	return getDimensions()[1];
    }

    public int getColumnElementsDistance() {
	return getDimensionElementsDistance(getDimensionGlobalIndex(1));
    }

    public int getRows() {
	return getDimensions()[0];
    }

    public int getRowElementsDistance() {
	return getDimensionElementsDistance(getDimensionGlobalIndex(0));
    }
}
