package com.github.neuralnetworks.architecture;


/**
 * Convolutional connection between layers (for 2d input data)
 */
public class Conv2DConnection extends ConnectionsImpl {

    /**
     * The list of filters to be used in the connection
     */
    protected float[] weights;
    protected int kernelColumns;
    protected int kernelRows;
    
    public Conv2DConnection(ConvGridLayer inputLayer, int kernelColumns, int kernelRows) {
	super(inputLayer, new ConvGridLayer());
	this.kernelColumns = kernelColumns;
	this.kernelRows = kernelRows;
	ConvGridLayer o = (ConvGridLayer) getOutputLayer();
	o.updateDimensions();
    }

    public Conv2DConnection(ConvGridLayer inputLayer, ConvGridLayer outputLayer) {
	super(inputLayer, outputLayer);
	updateDimensions();
    }

    /**
     * When some dimension changes in the output layer the weights array changes it's size
     */
    public void updateDimensions() {
	ConvGridLayer i = (ConvGridLayer) getInputLayer();
	ConvGridLayer o = (ConvGridLayer) getOutputLayer();
	kernelColumns = i.getColumns() % o.getColumns() + 1;
	kernelRows = i.getRows() % o.getRows() + 1;
	updateWeights();
    }

    protected void updateWeights() {
	ConvGridLayer i = (ConvGridLayer) getInputLayer();
	ConvGridLayer o = (ConvGridLayer) getOutputLayer();
	int totalWeights = kernelColumns * kernelRows * o.getFilters() * i.getFilters();
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

    public int getKernelColumns() {
        return kernelColumns;
    }

    public void setKernelColumns(int kernelColumns) {
        this.kernelColumns = kernelColumns;
        updateWeights();
	ConvGridLayer o = (ConvGridLayer) getOutputLayer();
	o.updateDimensions();
    }

    public int getKernelRows() {
        return kernelRows;
    }

    public void setKernelRows(int kernelRows) {
        this.kernelRows = kernelRows;
        updateWeights();
	ConvGridLayer o = (ConvGridLayer) getOutputLayer();
	o.updateDimensions();
    }
}
