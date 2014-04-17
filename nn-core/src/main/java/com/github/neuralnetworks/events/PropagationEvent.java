package com.github.neuralnetworks.events;

import java.util.EventObject;
import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;

/**
 * Triggered when a propagation step is finished
 */
public class PropagationEvent extends EventObject {

    private static final long serialVersionUID = 1L;

    private ValuesProvider results;
    private List<Connections> connections;
    private NeuralNetwork neuralNetwork;

    public PropagationEvent(Layer layer, List<Connections> connections, NeuralNetwork nn, ValuesProvider results) {
	super(layer);
	this.connections = connections;
	this.neuralNetwork = nn;
	this.results = results;
    }

    public ValuesProvider getResults() {
	return results;
    }
    
    public void setResults(ValuesProvider results) {
	this.results = results;
    }

    public Layer getLayer() {
	return (Layer) getSource();
    }

    public List<Connections> getConnections() {
        return connections;
    }

    public void setConnections(List<Connections> connections) {
        this.connections = connections;
    }

    public NeuralNetwork getNeuralNetwork() {
        return neuralNetwork;
    }

    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }
}