package com.github.neuralnetworks.architecture;

public class Pooling2DConnection extends ConnectionsImpl {

    private int poolingRegionRows;
    private int poolingRegionCols;

    public Pooling2DConnection(int poolingRegionRows, int poolingRegionCols, ConvGridLayer inputLayer, Layer outputLayer) {
	super(inputLayer, outputLayer);
	this.poolingRegionRows = poolingRegionRows;
	this.poolingRegionCols = poolingRegionCols;
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
}
