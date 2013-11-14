package com.github.neuralnetworks.architecture;

import java.util.List;

import com.github.neuralnetworks.util.UniqueList;

/**
 * Convolutional connection between layers
 */
public class ConvConnection extends ConnectionsImpl {

    protected List<Matrix> featureMapWeights;

    public ConvConnection(Layer inputLayer, Layer outputLayer) {
	super(inputLayer, outputLayer);
    }

    public List<Matrix> getFeatureMapWeights() {
	return featureMapWeights;
    }

    public void setFeatureMapWeights(List<Matrix> featureMapWeights) {
	this.featureMapWeights = featureMapWeights;
    }

    public void addFeatureMap(Matrix featureMap) {
	if (featureMapWeights ==  null) {
	    featureMapWeights = new UniqueList<>();
	}

	featureMapWeights.add(featureMap);
    }
}
