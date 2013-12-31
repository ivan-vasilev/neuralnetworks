package com.github.neuralnetworks.architecture;


/**
 * Three dimensional layer to be used for convolutional and subsampling connections.
 */
public class ConvGridLayer extends Layer {

    private static final long serialVersionUID = -4824165465883890932L;

    private int featureMapColumns;
    private int featureMapRows;
    private int filters;

    public ConvGridLayer() {
	super(0);
    }

    public ConvGridLayer(int featureMapRows, int featureMapColumns, int filters) {
	super(featureMapRows * featureMapColumns * filters);
	this.featureMapColumns = featureMapColumns;
	this.featureMapRows = featureMapRows;
	this.filters = filters;
    }

    public int getFeatureMapColumns() {
	return featureMapColumns;
    }

    public void setDimensions(int featureMaprows, int featureMapColumns, int filters) {
	this.featureMapRows = featureMaprows;
	this.featureMapColumns = featureMapColumns;
	this.filters = filters;

	updateNeuronCount();
    }

    public int getFeatureMapRows() {
	return featureMapRows;
    }

    public int getFeatureMapLength() {
	return featureMapRows * featureMapColumns;
    }

    public int getFilters() {
	return filters;
    }

    protected void updateNeuronCount() {
	setNeuronCount(featureMapRows * featureMapColumns * filters);
    }

    protected Conv2DConnection getInputConvConnection() {
	for (Connections c : getConnections()) {
	    if (c instanceof Conv2DConnection && c.getOutputLayer() == this) {
		return (Conv2DConnection) c;
	    }
	}

	return null;
    }
}
