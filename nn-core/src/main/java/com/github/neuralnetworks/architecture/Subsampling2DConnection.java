package com.github.neuralnetworks.architecture;

import com.github.neuralnetworks.calculation.ConnectionCalculator;

public class Subsampling2DConnection extends ConnectionsImpl {

    private int subsamplingRegionRows;
    private int subsamplingRegionCols;

    public Subsampling2DConnection(ConvGridLayer inputLayer, int subsamplingRegionRows, int subsamplingRegionCols, ConnectionCalculator outputConnectionCalculator) {
	super(inputLayer, new ConvGridLayer(0, 0, inputLayer.getFilters(), outputConnectionCalculator));
	
	this.subsamplingRegionRows = subsamplingRegionRows;
	this.subsamplingRegionCols = subsamplingRegionCols;
	updateOutputLayerDims();
    }

    public int getSubsamplingRegionRows() {
	return subsamplingRegionRows;
    }

    public void setSubsamplingRegionRows(int subsamplingRegionRows) {
	this.subsamplingRegionRows = subsamplingRegionRows;
    }

    public int getSubsamplingRegionCols() {
	return subsamplingRegionCols;
    }

    public void setSubsamplingRegionCols(int subsamplingRegionCols) {
	this.subsamplingRegionCols = subsamplingRegionCols;
    }

    protected void updateOutputLayerDims() {
	if (subsamplingRegionRows != 0 && subsamplingRegionCols != 0) {
	    ConvGridLayer inputLayer = (ConvGridLayer) getInputLayer();
	    ConvGridLayer outputLayer = (ConvGridLayer) getOutputLayer();
	    outputLayer.setRows(inputLayer.getRows() / subsamplingRegionRows);
	    outputLayer.setColumns(inputLayer.getColumns() / subsamplingRegionCols);
	}
    }
}
