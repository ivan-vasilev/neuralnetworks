package com.github.neuralnetworks.architecture;

import java.util.List;

import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Convolutional connection between layers (for 2d input data)
 */
public class Conv2DConnection extends ConnectionsImpl {

    protected List<Matrix> featureMaps;

    public Conv2DConnection(ConvGridLayer inputLayer, ConnectionCalculator convCalculator) {
	super(inputLayer, new ConvGridLayer(0, 0, 0, convCalculator));
    }

    public List<Matrix> getFeatureMaps() {
	return featureMaps;
    }

    public void setFeatureMaps(List<Matrix> featureMaps) {
	this.featureMaps = featureMaps;
    }

    public void addFeatureMap(Matrix featureMap) {
	if (featureMaps == null) {
	    featureMaps = new UniqueList<>();
	}

	for (Matrix m : featureMaps) {
	    if (featureMap.getColumns() != m.getColumns() || featureMap.getRows() != m.getRows()) {
		throw new IllegalArgumentException();
	    }
	}

	featureMaps.add(featureMap);

	ConvGridLayer inputLayer = (ConvGridLayer) getInputLayer();
	ConvGridLayer outputLayer = (ConvGridLayer) getOutputLayer();
	outputLayer.setColumns(inputLayer.getColumns() - inputLayer.getColumns() % featureMap.getColumns());
	outputLayer.setRows(inputLayer.getRows() - inputLayer.getRows() % featureMap.getRows());
	outputLayer.setFeatureMaps(featureMaps.size());
    }

    public void removeFeatureMap(Matrix featureMap) {
	if (featureMaps != null) {
	    featureMaps.remove(featureMap);
	    ConvGridLayer l = (ConvGridLayer) getOutputLayer();
	    l.setFeatureMaps(featureMaps.size());
	}
    }
}
