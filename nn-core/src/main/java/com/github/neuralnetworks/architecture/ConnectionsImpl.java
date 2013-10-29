package com.github.neuralnetworks.architecture;

/**
 * this abstract class serves as a base for all weight matrices
 * 
 * @author hok
 * 
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

    @Override
    public Layer getOutputLayer() {
	return outputLayer;
    }

    @Override
    public int compareTo(ConnectionsImpl o) {
	return this.toString().compareTo(o.toString());
    }
}
