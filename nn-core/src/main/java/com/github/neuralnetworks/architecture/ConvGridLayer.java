package com.github.neuralnetworks.architecture;

import com.github.neuralnetworks.calculation.ConnectionCalculator;

/**
 * 
 * 2 dimensional layer
 * 
 */
public class ConvGridLayer extends Layer {

    private static final long serialVersionUID = -4824165465883890932L;

    private int columns;
    private int rows;
    private int featureMaps;

    public ConvGridLayer(ConnectionCalculator connectionCalculator) {
	super(0, connectionCalculator);
    }

    public ConvGridLayer(int rows, int columns, int featureMaps, ConnectionCalculator connectionCalculator) {
	super(rows * columns * featureMaps, connectionCalculator);
	this.columns = columns;
	this.rows = rows;
	this.featureMaps = featureMaps;
    }

    public ConvGridLayer(int rows, int columns, int featureMaps) {
	super(rows * columns * featureMaps);
	this.columns = columns;
	this.rows = rows;
	this.featureMaps = featureMaps;
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

    public int getFeatureMaps() {
	return featureMaps;
    }

    public void setFeatureMaps(int featureMaps) {
	this.featureMaps = featureMaps;
	updateNeuronCount();
    }

    protected void updateNeuronCount() {
	setNeuronCount(rows * columns * featureMaps);
    }
}
