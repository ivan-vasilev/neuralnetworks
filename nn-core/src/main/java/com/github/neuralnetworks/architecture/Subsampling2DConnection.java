package com.github.neuralnetworks.architecture;

/**
 * Subsampling connections. Contains information about the size of the
 * subsampling region
 */
public class Subsampling2DConnection extends ConnectionsImpl {

    public Subsampling2DConnection(ConvGridLayer inputLayer, int subsamplingRegionRows, int subsamplingRegionCols) {
	super(inputLayer, new ConvGridLayer());
	setDimensions(subsamplingRegionRows, subsamplingRegionCols);
    }

    public Subsampling2DConnection(ConvGridLayer inputLayer, ConvGridLayer outputLayer) {
	super(inputLayer, outputLayer);
    }

    public void setDimensions(int subsamplingRegionRows, int subsamplingRegionCols) {
	ConvGridLayer inputLayer = (ConvGridLayer) getInputLayer();
	ConvGridLayer outputLayer = (ConvGridLayer) getOutputLayer();
	outputLayer.setDimensions(inputLayer.getFeatureMapRows() / subsamplingRegionRows, inputLayer.getFeatureMapColumns() / subsamplingRegionCols, inputLayer.getFilters());
    }

    public int getSubsamplingRegionRows() {
	ConvGridLayer inputLayer = (ConvGridLayer) getInputLayer();
	ConvGridLayer outputLayer = (ConvGridLayer) getOutputLayer();
	return inputLayer.getFeatureMapRows() / outputLayer.getFeatureMapRows();
    }

    public int getSubsamplingRegionCols() {
	ConvGridLayer inputLayer = (ConvGridLayer) getInputLayer();
	ConvGridLayer outputLayer = (ConvGridLayer) getOutputLayer();
	return inputLayer.getFeatureMapColumns() / outputLayer.getFeatureMapColumns();
    }

    public int getSubsamplingRegionLength() {
	return getSubsamplingRegionRows() * getSubsamplingRegionCols();
    }
}
