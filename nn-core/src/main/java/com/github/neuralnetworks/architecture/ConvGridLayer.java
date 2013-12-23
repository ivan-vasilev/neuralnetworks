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
    }

    public int getRows() {
	return rows;
    }

    public void setRows(int rows) {
	this.rows = rows;
	updateNeuronCount();
    }

    public int getFilters() {
	return filters;
    }

    public void setFilters(int filters) {
	this.filters = filters;
	updateNeuronCount();
    }

    protected void updateNeuronCount() {
	setNeuronCount(rows * columns * filters);
    }
}
