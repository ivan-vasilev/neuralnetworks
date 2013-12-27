package com.github.neuralnetworks.architecture;

/**
 * Convolutional connection between layers (for 2d input data)
 */
public class Conv2DConnection extends ConnectionsImpl {

    /**
     * The list of filters to be used in the connection
     */
    protected float[] weights;
    
    public Conv2DConnection(ConvGridLayer inputLayer, int kernelColumns, int kernelRows) {
	super(inputLayer, new ConvGridLayer());
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
	int totalWeights = getKernelColumns() * getKernelRows() * o.getFilters() * i.getFilters();
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
        ConvGridLayer i = (ConvGridLayer) getInputLayer();
	ConvGridLayer o = (ConvGridLayer) getOutputLayer();
	return i.getColumns() % o.getColumns() + 1;
    }

    public void setKernelColumns(int kernelColumns) {
        ConvGridLayer i = (ConvGridLayer) getInputLayer();
	ConvGridLayer o = (ConvGridLayer) getOutputLayer();
	o.setColumns(i.getColumns() - i.getColumns() % kernelColumns);
    }

    public int getKernelRows() {
        ConvGridLayer i = (ConvGridLayer) getInputLayer();
	ConvGridLayer o = (ConvGridLayer) getOutputLayer();
	return i.getRows() % o.getRows() + 1;
    }

    public void setKernelRows(int kernelRows) {
        ConvGridLayer i = (ConvGridLayer) getInputLayer();
	ConvGridLayer o = (ConvGridLayer) getOutputLayer();
	o.setRows(i.getRows() - i.getRows() % kernelRows);
    }
}
