package com.github.neuralnetworks.architecture;


/**
 * Subsampling connections. Contains information about the size of the subsampling region
 */
public class Subsampling2DConnection extends ConnectionsImpl {

    /**
     * subsampling region rows
     */
    private int subsamplingRegionRows;

    /**
     * subsampling region columns
     */
    private int subsamplingRegionCols;

    public Subsampling2DConnection(ConvGridLayer inputLayer, int subsamplingRegionRows, int subsamplingRegionCols) {
	super(inputLayer, new ConvGridLayer(0, 0, inputLayer.getFilters()));
	
	this.subsamplingRegionRows = subsamplingRegionRows;
	this.subsamplingRegionCols = subsamplingRegionCols;

	// update the dimensions of the output layer
	updateOutputLayerDimmensions();
    }

    public int getSubsamplingRegionRows() {
	return subsamplingRegionRows;
    }

    public void setSubsamplingRegionRows(int subsamplingRegionRows) {
	this.subsamplingRegionRows = subsamplingRegionRows;
	updateOutputLayerDimmensions();
    }

    public int getSubsamplingRegionCols() {
	return subsamplingRegionCols;
    }

    public void setSubsamplingRegionCols(int subsamplingRegionCols) {
	this.subsamplingRegionCols = subsamplingRegionCols;
	updateOutputLayerDimmensions();
    }

    /**
     * When the size of the subsampling region is changed, then the neuron count in the output layer is also changed
     */
    protected void updateOutputLayerDimmensions() {
	if (subsamplingRegionRows != 0 && subsamplingRegionCols != 0) {
	    ConvGridLayer inputLayer = (ConvGridLayer) getInputLayer();
	    ConvGridLayer outputLayer = (ConvGridLayer) getOutputLayer();
	    outputLayer.setRows(inputLayer.getFeatureMapRows() / subsamplingRegionRows);
	    outputLayer.setFeatureMapColumns(inputLayer.getFeatureMapColumns() / subsamplingRegionCols);
	}
    }
}
