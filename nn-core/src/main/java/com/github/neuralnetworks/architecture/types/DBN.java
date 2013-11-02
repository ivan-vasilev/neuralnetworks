package com.github.neuralnetworks.architecture.types;

import java.util.List;

import com.github.neuralnetworks.architecture.DeepNeuralNetwork;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.util.UniqueList;

public class DBN extends NeuralNetworkImpl implements DeepNeuralNetwork<RBM> {

    protected List<RBM> neuralNetworks;

    public DBN() {
	super();
	this.neuralNetworks = new UniqueList<>();
    }

    public DBN addLayer(Layer layer, boolean addBias) {
	Layer currentOutputLayer = getOutputLayer();
	if (addLayer(layer) && getLayers().size() > 0) {
	    neuralNetworks.add(new RBM(currentOutputLayer, layer, false, addBias));
	}

	return this;
    }

    @Override
    public List<RBM> getNeuralNetworks() {
	return neuralNetworks;
    }
}
