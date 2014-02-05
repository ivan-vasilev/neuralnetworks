package com.github.neuralnetworks.architecture;

/**
 * Convolutional connection between layers (for 2d input data)
 */
public class Conv2DConnection extends ConnectionsImpl {

    /**
     * The list of filters to be used in the connection
     */
    protected float[] weights;
    
    public Conv2DConnection(ConvGridLayer inputLayer, int kernelRows, int kernelColumns, int filters) {
	super(inputLayer, new ConvGridLayer());
	setDimensions(kernelRows, kernelColumns, filters);
	updateDimensions();
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

    public int getKernelRows() {
        ConvGridLayer i = (ConvGridLayer) getInputLayer();
	ConvGridLayer o = (ConvGridLayer) getOutputLayer();
	return i.getFeatureMapRows() % o.getFeatureMapRows() + 1;
    }

    public int getKernelColumns() {
        ConvGridLayer i = (ConvGridLayer) getInputLayer();
	ConvGridLayer o = (ConvGridLayer) getOutputLayer();
	return i.getFeatureMapColumns() % o.getFeatureMapColumns() + 1;
    }

    public void setDimensions(int kernelRows, int kernelColumns, int filters) {
        ConvGridLayer i = (ConvGridLayer) getInputLayer();
	ConvGridLayer o = (ConvGridLayer) getOutputLayer();
	o.setDimensions(i.getFeatureMapRows() - kernelRows + 1, i.getFeatureMapColumns() - kernelColumns + 1, filters);
    }
}
