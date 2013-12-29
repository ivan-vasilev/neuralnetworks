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

    public void setFeatureMapColumns(int featureMapColumns) {
	this.featureMapColumns = featureMapColumns;
	updateNeuronCount();
	getInputConnection().updateDimensions();
    }

    public int getFeatureMapRows() {
	return featureMapRows;
    }

    public void setFeatureMapRows(int featureMaprows) {
	this.featureMapRows = featureMaprows;
	updateNeuronCount();
	getInputConnection().updateDimensions();
    }

    public int getFeatureMapLength() {
	return featureMapRows * featureMapColumns;
    }

    public int getFilters() {
	return filters;
    }

    public void setFilters(int filters) {
	this.filters = filters;
	updateNeuronCount();
	getInputConnection().updateDimensions();
    }
    
    public void updateDimensions() {
	for (Connections c : getConnections()) {
	    Conv2DConnection con = (Conv2DConnection) c;
	    if (con.getOutputLayer() == this) {
		ConvGridLayer input = (ConvGridLayer) con.getInputLayer();
		setFeatureMapRows(input.getFeatureMapRows() - input.getFeatureMapRows() % con.getKernelRows());
		setFeatureMapColumns(input.getFeatureMapColumns() - input.getFeatureMapColumns() % con.getKernelColumns());
		break;
	    }
	}
    }

    protected void updateNeuronCount() {
	setNeuronCount(featureMapRows * featureMapColumns * filters);
    }

    protected Conv2DConnection getInputConnection() {
	for (Connections c : getConnections()) {
	    if (c instanceof Conv2DConnection && c.getOutputLayer() == this) {
		return (Conv2DConnection) c;
	    }
	}

	return null;
    }
}
