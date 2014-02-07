package com.github.neuralnetworks.architecture;

/**
 * Three dimensional layer to be used for convolutional and subsampling
 * connections.
 */
public class ConvGridLayer extends Layer {

    private static final long serialVersionUID = -4824165465883890932L;

    private int featureMapColumns;
    private int featureMapRows;
    private int filters;
    private boolean isBias;

    public ConvGridLayer() {
	super();
    }

    public ConvGridLayer(int featureMapRows, int featureMapColumns, int filters) {
	super();
	this.featureMapColumns = featureMapColumns;
	this.featureMapRows = featureMapRows;
	this.filters = filters;
    }

    public ConvGridLayer(int featureMapRows, int featureMapColumns, int filters, boolean isBias) {
	this(featureMapRows, featureMapColumns, filters);
	this.isBias = isBias;
    }

    public int getFeatureMapColumns() {
	return featureMapColumns;
    }

    public void setDimensions(int featureMaprows, int featureMapColumns, int filters) {
	this.featureMapRows = featureMaprows;
	this.featureMapColumns = featureMapColumns;
	this.filters = filters;
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

    public boolean isBias() {
	return isBias;
    }

    public void setBias(boolean isBias) {
	this.isBias = isBias;
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
