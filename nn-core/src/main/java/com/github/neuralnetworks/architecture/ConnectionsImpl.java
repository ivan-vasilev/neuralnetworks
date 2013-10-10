package com.github.neuralnetworks.architecture;

/**
 * this abstract class serves as a base for all weight matrices
 * 
 * @author hok
 * 
 */
public abstract class ConnectionsImpl implements Connections {

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

	inputLayer.addConnectionGraph(this);
	outputLayer.addConnectionGraph(this);
    }

    @Override
    public Layer getInputLayer() {
	return inputLayer;
    }

    @Override
    public Layer getOutputLayer() {
	return outputLayer;
    }
}
