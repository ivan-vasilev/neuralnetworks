package com.github.neuralnetworks.architecture;

/**
 * Convolutional connection between layers (for 2d input data)
 */
public class Conv2DConnection extends ConnectionsImpl {

    /**
     * The list of filters to be used in the connection
     */
    protected float[] weights;
    protected int inputFeatureMapColumns;
    protected int inputFeatureMapRows;
    protected int inputFilters;
    protected int kernelRows;
    protected int kernelColumns;
    protected int outputFilters;
    protected int stride;

    public Conv2DConnection(Layer inputLayer, Layer outputLayer, int inputFeatureMapColumns, int inputFeatureMapRows, int inputFilters, int kernelRows, int kernelColumns, int outputFilters, int stride) {
	super(inputLayer, outputLayer);
	this.inputFeatureMapColumns = inputFeatureMapColumns;
	this.inputFeatureMapRows = inputFeatureMapRows;
	this.inputFilters = inputFilters;
	this.outputFilters = outputFilters;
	this.kernelRows = kernelRows;
	this.kernelColumns = kernelColumns;
	this.stride = stride;
	updateDimensions();
    }

    /**
     * When some dimension changes in the output layer the weights array changes it's size
     */
    public void updateDimensions() {
	int totalWeights = getKernelColumns() * getKernelRows() * outputFilters * inputFilters;
	if (weights == null || weights.length != totalWeights) {
	    weights = new float[totalWeights];
	}
    }

    public float[] getWeights() {
	return weights;
    }

    public void setWeights(float[] weights) {
	this.weights = weights;
    }

    public int getKernelRows() {
	return kernelRows;
    }

    public int getKernelColumns() {
	return kernelColumns;
    }

    @Override
    public int getInputUnitCount() {
	return inputFeatureMapRows * inputFeatureMapColumns * inputFilters;
    }

    @Override
    public int getOutputUnitCount() {
	return getOutputFeatureMapLength() * outputFilters;
    }

    public int getInputFeatureMapColumns() {
        return inputFeatureMapColumns;
    }

    public void setInputFeatureMapColumns(int inputFeatureMapColumns) {
        this.inputFeatureMapColumns = inputFeatureMapColumns;
    }

    public int getInputFeatureMapRows() {
        return inputFeatureMapRows;
    }

    public void setInputFeatureMapRows(int inputFeatureMapRows) {
        this.inputFeatureMapRows = inputFeatureMapRows;
    }

    public int getInputFeatureMapLength() {
	return inputFeatureMapRows * inputFeatureMapColumns;
    }
    
    public int getOutputFeatureMapLength() {
	return getOutputFeatureMapRows() * getOutputFeatureMapColumns();
    }

    public int getInputFilters() {
        return inputFilters;
    }

    public void setInputFilters(int inputFilters) {
        this.inputFilters = inputFilters;
    }

    public int getOutputFeatureMapColumns() {
        return (inputFeatureMapRows - kernelRows) / stride + 1;
    }

    public int getOutputFeatureMapRows() {
        return (inputFeatureMapColumns - kernelColumns) / stride + 1;
    }

    public int getOutputFilters() {
        return outputFilters;
    }

    public void setOutputFilters(int outputFilters) {
        this.outputFilters = outputFilters;
    }

    public int getStride() {
        return stride;
    }

    public void setStride(int stride) {
        this.stride = stride;
    }
}
