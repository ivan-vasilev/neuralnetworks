package com.github.neuralnetworks.architecture;

/**
 * Subsampling connections. Contains information about the size of the
 * subsampling region
 */
public class Subsampling2DConnection extends ConnectionsImpl {

    private static final long serialVersionUID = 1L;

    protected int inputFeatureMapColumns;
    protected int inputFeatureMapRows;
    protected int outputFeatureMapColumns;
    protected int outputFeatureMapRows;
    protected int filters;

    public Subsampling2DConnection(Layer inputLayer, Layer outputLayer, int inputFeatureMapColumns, int inputFeatureMapRows, int subsamplingRegionRows, int subsamplingRegionCols, int filters) {
	super(inputLayer, outputLayer);
	this.inputFeatureMapColumns = inputFeatureMapColumns;
	this.inputFeatureMapRows = inputFeatureMapRows;
	this.filters = filters;
	setDimensions(subsamplingRegionRows, subsamplingRegionCols);
    }

    public void setDimensions(int subsamplingRegionRows, int subsamplingRegionCols) {
	setOutputFeatureMapRows(inputFeatureMapRows / subsamplingRegionRows);
	setOutputFeatureMapColumns(inputFeatureMapColumns / subsamplingRegionCols);
    }

    public int getSubsamplingRegionRows() {
	return inputFeatureMapRows / outputFeatureMapRows;
    }

    public int getSubsamplingRegionCols() {
	return inputFeatureMapColumns / outputFeatureMapColumns;
    }

    public int getSubsamplingRegionLength() {
	return getSubsamplingRegionRows() * getSubsamplingRegionCols();
    }

    @Override
    public int getInputUnitCount() {
	return inputFeatureMapRows * inputFeatureMapColumns * filters;
    }

    @Override
    public int getOutputUnitCount() {
	return outputFeatureMapRows * outputFeatureMapColumns * filters;
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

    public int getFilters() {
        return filters;
    }

    public void setFilters(int filters) {
        this.filters = filters;
    }

    public int getInputFeatureMapLength() {
	return inputFeatureMapRows * inputFeatureMapColumns;
    }
    
    public int getOutputFeatureMapLength() {
	return outputFeatureMapRows * outputFeatureMapColumns;
    }
}
