package com.github.neuralnetworks.architecture;

import com.github.neuralnetworks.util.Tensor;

/**
 * Convolutional connection between layers (for 2d input data)
 */
public class Conv2DConnection extends ConnectionsImpl {

    private static final long serialVersionUID = 1L;

    /**
     * The list of filters to be used in the connection
     */
    protected Tensor wights;
    protected int inputFeatureMapColumns;
    protected int inputFeatureMapRows;
    protected int stride;

    public Conv2DConnection(Layer inputLayer, Layer outputLayer, int inputFeatureMapColumns, int inputFeatureMapRows, int inputFilters, int kernelRows, int kernelColumns, int outputFilters, int stride) {
	super(inputLayer, outputLayer);
	this.inputFeatureMapColumns = inputFeatureMapColumns;
	this.inputFeatureMapRows = inputFeatureMapRows;
	this.stride = stride;
	this.wights = new Tensor(outputFilters, inputFilters, kernelRows, kernelColumns);
    }

    public Tensor getWeights() {
	return wights;
    }

    public int getKernelRows() {
	return wights.getDimensionLength(2);
    }

    public int getKernelColumns() {
	return wights.getDimensionLength(3);
    }

    @Override
    public int getInputUnitCount() {
	return inputFeatureMapRows * inputFeatureMapColumns * wights.getDimensionLength(1);
    }

    @Override
    public int getOutputUnitCount() {
	return getOutputFeatureMapLength() * wights.getDimensionLength(0);
    }

    public int getInputFeatureMapColumns() {
        return inputFeatureMapColumns;
    }

    public int getInputFeatureMapRows() {
        return inputFeatureMapRows;
    }

    public int getInputFeatureMapLength() {
	return inputFeatureMapRows * inputFeatureMapColumns;
    }
    
    public int getOutputFeatureMapLength() {
	return getOutputFeatureMapRows() * getOutputFeatureMapColumns();
    }

    public int getInputFilters() {
        return wights.getDimensionLength(1);
    }

    public int getOutputFeatureMapColumns() {
        return (inputFeatureMapRows - wights.getDimensionLength(2)) / stride + 1;
    }

    public int getOutputFeatureMapRows() {
        return (inputFeatureMapColumns - wights.getDimensionLength(3)) / stride + 1;
    }

    public int getOutputFilters() {
        return wights.getDimensionLength(0);
    }

    public int getStride() {
        return stride;
    }
}
