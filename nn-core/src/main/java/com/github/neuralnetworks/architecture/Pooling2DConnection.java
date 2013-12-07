package com.github.neuralnetworks.architecture;

import com.github.neuralnetworks.calculation.ConnectionCalculator;

public class Pooling2DConnection extends ConnectionsImpl {

    private int poolingRegionRows;
    private int poolingRegionCols;

    public Pooling2DConnection(ConvGridLayer inputLayer, int poolingRegionRows, int poolingRegionCols, ConnectionCalculator outputConnectionCalculator) {
	super(inputLayer, new ConvGridLayer(0, 0, inputLayer.getFilters(), outputConnectionCalculator));
	
	this.poolingRegionRows = poolingRegionRows;
	this.poolingRegionCols = poolingRegionCols;
	updateOutputLayerDims();
    }

    public int getPoolingRegionRows() {
	return poolingRegionRows;
    }

    public void setPoolingRegionRows(int poolingRegionRows) {
	this.poolingRegionRows = poolingRegionRows;
    }

    public int getPoolingRegionCols() {
	return poolingRegionCols;
    }

    public void setPoolingRegionCols(int poolingRegionCols) {
	this.poolingRegionCols = poolingRegionCols;
    }

    protected void updateOutputLayerDims() {
	ConvGridLayer inputLayer = (ConvGridLayer) getInputLayer();
	ConvGridLayer outputLayer = (ConvGridLayer) getOutputLayer();
	outputLayer.setRows(inputLayer.getRows() - inputLayer.getRows() % poolingRegionRows);
	outputLayer.setColumns(inputLayer.getColumns() - inputLayer.getColumns() % poolingRegionCols);
    }
}
