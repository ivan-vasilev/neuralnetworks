package com.github.neuralnetworks.architecture;

import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.TensorFactory;

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
	this.wights = TensorFactory.tensor(outputFilters, inputFilters, kernelRows, kernelColumns);
    }

    public Tensor getWeights() {
	return wights;
    }

    public int getKernelRows() {
	return wights.getDimensions()[2];
    }

    public int getKernelColumns() {
	return wights.getDimensions()[3];
    }

    @Override
    public int getInputUnitCount() {
	return inputFeatureMapRows * inputFeatureMapColumns * wights.getDimensions()[1];
    }

    @Override
    public int getOutputUnitCount() {
	return getOutputFeatureMapLength() * wights.getDimensions()[0];
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
        return wights.getDimensions()[1];
    }

    public int getOutputFeatureMapColumns() {
        return (inputFeatureMapRows - wights.getDimensions()[2]) / stride + 1;
    }

    public int getOutputFeatureMapRows() {
        return (inputFeatureMapColumns - wights.getDimensions()[3]) / stride + 1;
    }

    public int getOutputFilters() {
        return wights.getDimensions()[0];
    }

    public int getStride() {
        return stride;
    }
}
