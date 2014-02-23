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
    protected int outputFeatureMapColumns;
    protected int outputFeatureMapRows;
    protected int outputFilters;

    public Conv2DConnection(Layer inputLayer, Layer outputLayer, int inputFeatureMapColumns, int inputFeatureMapRows, int inputFilters, int kernelRows, int kernelColumns, int outputFilters) {
	super(inputLayer, outputLayer);
	this.inputFeatureMapColumns = inputFeatureMapColumns;
	this.inputFeatureMapRows = inputFeatureMapRows;
	this.inputFilters = inputFilters;
	this.outputFilters = outputFilters;
	setOutputDimensions(kernelRows, kernelColumns);
    }

    public void setOutputDimensions(int kernelRows, int kernelColumns) {
	outputFeatureMapRows = inputFeatureMapRows - kernelRows + 1;
	outputFeatureMapColumns = inputFeatureMapColumns - kernelColumns + 1;
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
	return inputFeatureMapRows % outputFeatureMapRows + 1;
    }

    public int getKernelColumns() {
	return inputFeatureMapColumns % outputFeatureMapColumns + 1;
    }

    @Override
    public int getInputUnitCount() {
	return inputFeatureMapRows * inputFeatureMapColumns * inputFilters;
    }

    @Override
    public int getOutputUnitCount() {
	return outputFeatureMapRows * outputFeatureMapColumns * outputFilters;
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
	return outputFeatureMapRows * outputFeatureMapColumns;
    }

    public int getInputFilters() {
        return inputFilters;
    }

    public void setInputFilters(int inputFilters) {
        this.inputFilters = inputFilters;
    }

    public int getOutputFeatureMapColumns() {
        return outputFeatureMapColumns;
    }

    public void setOutputFeatureMapColumns(int outputFeatureMapColumns) {
        this.outputFeatureMapColumns = outputFeatureMapColumns;
    }

    public int getOutputFeatureMapRows() {
        return outputFeatureMapRows;
    }

    public void setOutputFeatureMapRows(int outputFeatureMapRows) {
        this.outputFeatureMapRows = outputFeatureMapRows;
    }

    public int getOutputFilters() {
        return outputFilters;
    }

    public void setOutputFilters(int outputFilters) {
        this.outputFilters = outputFilters;
    }
}
