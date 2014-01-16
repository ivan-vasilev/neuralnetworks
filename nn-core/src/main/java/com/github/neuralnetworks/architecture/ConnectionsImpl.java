package com.github.neuralnetworks.architecture;

import java.util.List;
import java.util.Set;

import com.github.neuralnetworks.calculation.LayerCalculator;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Base implementation of Connections
 */
public abstract class ConnectionsImpl implements Connections, Comparable<ConnectionsImpl> {

    /**
     * input layer of neurons
     */
    protected Layer inputLayer;

    /**
     * output layer
     */
    protected Layer outputLayer;

    public ConnectionsImpl(Layer inputLayer, Layer outputLayer) {
	super();
	this.inputLayer = inputLayer;
	this.outputLayer = outputLayer;

	inputLayer.addConnection(this);
	outputLayer.addConnection(this);
    }

    @Override
    public Layer getInputLayer() {
	return inputLayer;
    }
    
    public void setInputLayer(Layer inputLayer) {
	this.inputLayer = inputLayer;
    }

    @Override
    public Layer getOutputLayer() {
	return outputLayer;
    }

    public void setOutputLayer(Layer outputLayer) {
        this.outputLayer = outputLayer;
    }

    @Override
    public Layer getDataOutputLayer() {
	return getOutputLayer();
    }

    @Override
    public Set<Layer> getLayers() {
	Set<Layer> result = new UniqueList<Layer>();
	result.add(getInputLayer());
	result.add(getOutputLayer());
	return result;
    }

    @Override
    public List<Connections> getConnections() {
	List<Connections> result = new UniqueList<Connections>();
	result.add(this);
	return result;
    }

    @Override
    public LayerCalculator getLayerCalculator() {
	return null;
    }

    @Override
    public int compareTo(ConnectionsImpl o) {
	return this.toString().compareTo(o.toString());
    }
}
