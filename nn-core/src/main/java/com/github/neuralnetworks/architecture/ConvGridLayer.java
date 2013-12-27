package com.github.neuralnetworks.architecture;


/**
 * Three dimensional layer to be used for convolutional and subsampling connections.
 */
public class ConvGridLayer extends Layer {

    private static final long serialVersionUID = -4824165465883890932L;

    private int columns;
    private int rows;
    private int filters;

    public ConvGridLayer() {
	super(0);
    }

    public ConvGridLayer(int rows, int columns, int filters) {
	super(rows * columns * filters);
	this.columns = columns;
	this.rows = rows;
	this.filters = filters;
    }

    public int getColumns() {
	return columns;
    }

    public void setColumns(int columns) {
	this.columns = columns;
	updateNeuronCount();
	getInputConnection().updateDimensions();
    }

    public int getRows() {
	return rows;
    }

    public void setRows(int rows) {
	this.rows = rows;
	updateNeuronCount();
	getInputConnection().updateDimensions();
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
		setRows(input.getRows() - input.getRows() % con.getKernelRows());
		setColumns(input.getColumns() - input.getColumns() % con.getKernelColumns());
		break;
	    }
	}
    }

    protected void updateNeuronCount() {
	setNeuronCount(rows * columns * filters);
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
