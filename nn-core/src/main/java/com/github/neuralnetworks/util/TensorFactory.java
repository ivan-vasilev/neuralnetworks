package com.github.neuralnetworks.util;

import java.util.stream.IntStream;

public class TensorFactory {

    @SuppressWarnings("unchecked")
    public static <T extends Tensor> T tensor(int... dimensions) {
	float[] elements = new float[IntStream.of(dimensions).reduce(1, (a, b) -> a * b)];
	int[][] dimensionsLimit = new int[2][dimensions.length];
	IntStream.range(0, dimensions.length).forEach(i -> dimensionsLimit[1][i] = dimensions[i] - 1);

	T result = null;
	if (dimensions.length == 2) {
	    result = (T) new Matrix(0, elements, dimensions, dimensionsLimit);
	} else {
	    result = (T) new Tensor(0, elements, dimensions, dimensionsLimit);
	}

	return result;
    }

    @SuppressWarnings("unchecked")
    public static <T extends Tensor> T tensor(float[] elements, int... dimensions) {
	int[][] dimensionsLimit = new int[2][dimensions.length];
	IntStream.range(0, dimensions.length).forEach(i -> dimensionsLimit[1][i] = dimensions[i] - 1);

	T result = null;
	if (dimensions.length == 2) {
	    result = (T) new Matrix(0, elements, dimensions, dimensionsLimit);
	} else {
	    result = (T) new Tensor(0, elements, dimensions, dimensionsLimit);
	}

	return result;
    }

    @SuppressWarnings("unchecked")
    public static <T extends Tensor> T tensor(Tensor parent, int[][] dimensionsLimit) {
	T result = null;

	long dimensions = IntStream.range(0, dimensionsLimit[0][0]).filter(i -> dimensionsLimit[0][i] != dimensionsLimit[1][i]).count();

	if (dimensions <= 2) {
	    result = (T) new Matrix(parent, dimensionsLimit);
	} else {
	    result = (T) new Tensor(parent, dimensionsLimit);
	}

	return result;
    }

    /**
     * @param copy
     * @return new tensor with the same dimensions
     */
    public static <T extends Tensor> T tensor(Tensor copy) {
	return tensor(copy.getDimensions());
    }

    /**
     * Simplified construction of matrix using values
     * @param elements
     * @return Matrix
     */
    public static Matrix matrix(float[][] elements) {
	Matrix result = tensor(elements[0].length, elements.length);
	IntStream.range(0, elements.length).forEach(i -> IntStream.range(0, elements[i].length).forEach(j -> {
	    result.set(elements[i][j], j, i);
	}));

	return result;
    }

    public static Matrix matrix(float[] elements, int columns) {
	return tensor(elements, elements.length / columns, columns);
    }
}
