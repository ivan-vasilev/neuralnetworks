package com.github.neuralnetworks.calculation;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.Util;

/**
 * Layer strategy for calculating specific layer
 */
public class TargetLayerOrderStrategy implements LayerOrderStrategy {

    private static final long serialVersionUID = 1L;

    private NeuralNetwork neuralNetwork;
    private Layer targetLayer;
    private Set<Layer> calculatedLayers;

    public TargetLayerOrderStrategy(NeuralNetwork neuralNetwork, Layer targetLayer, Set<Layer> calculatedLayers) {
	super();
	this.neuralNetwork = neuralNetwork;
	this.targetLayer = targetLayer;
	this.calculatedLayers = calculatedLayers;
    }

    @Override
    public List<ConnectionCandidate> order() {
	List<ConnectionCandidate> cc = new ArrayList<ConnectionCandidate>();

	orderConnections(neuralNetwork, targetLayer, calculatedLayers, new UniqueList<Layer>(), cc);

	return cc;
    }

    /**
     * Calculates single layer based on the network graph
     * 
     * For example the feedforward part of the backpropagation algorithm the initial parameters would be:
     * "currentLayer" will be the output layer of the network
     * "results" will contain only one entry for the input layer of the network - this is the training example
     * "calculatedLayers" will contain only one entry - the input layer
     * "inProgressLayers" will be empty
     * 
     * In the backpropagation part the initial parameters would be:
     * "currentLayer" will be the input layer of the network
     * "results" will contain only one entry for the output layer of the network - this is the calculated error derivative between the result of the network and the target value
     * "calculatedLayers" will contain only one entry - the output layer
     * "inProgressLayers" will be empty
     * 
     * This allows for single code to be used for the whole backpropagation, but also for RBMs, autoencoders, etc
     * 
     * @param neuralNetwork - the neural network.
     * @param calculatedLayers - layers that are fully calculated - the results for these layers can be used for calculating other parts of the network
     * @param inProgressLayers - layers which are currently calculated, but are not yet finished - not all connections to the layer are calculated and the result of the propagation through this layer cannot be used for another calculations
     * @param calculateCandidates - order of calculation
     * @param currentLayer - the layer which is currently being calculated.
     * @return
     */
    protected boolean orderConnections(NeuralNetwork neuralNetwork, Layer currentLayer, Set<Layer> calculatedLayers, Set<Layer> inProgressLayers, List<ConnectionCandidate> calculateCandidates) {
	boolean result = false;

	if (calculatedLayers.contains(currentLayer) || Util.isBias(currentLayer)) {
	    result = true;
	} else if (!inProgressLayers.contains(currentLayer)) {
	    inProgressLayers.add(currentLayer);
	    List<ConnectionCandidate> currentCandidates = new ArrayList<ConnectionCandidate>();

	    boolean hasNoBiasConnections = false;
	    for (Connections c : currentLayer.getConnections(neuralNetwork)) {
		Layer opposite = Util.getOppositeLayer(c, currentLayer);
		if (orderConnections(neuralNetwork, opposite, calculatedLayers, inProgressLayers, calculateCandidates)) {
		    currentCandidates.add(new ConnectionCandidate(c, currentLayer));

		    if (!Util.isBias(opposite)) {
			hasNoBiasConnections = true;
		    }
		}
	    }

	    if (currentCandidates.size() > 0 && hasNoBiasConnections)  {
		calculateCandidates.addAll(currentCandidates);
		result = true;
	    }

	    inProgressLayers.remove(currentLayer);
	    calculatedLayers.add(currentLayer);
	}

	return result;
    }

    public NeuralNetwork getNeuralNetwork() {
	return neuralNetwork;
    }

    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
	this.neuralNetwork = neuralNetwork;
    }

    public Layer getTargetLayer() {
	return targetLayer;
    }

    public void setTargetLayer(Layer targetLayer) {
	this.targetLayer = targetLayer;
    }

    public Set<Layer> getCalculatedLayers() {
        return calculatedLayers;
    }

    public void setCalculatedLayers(Set<Layer> calculatedLayers) {
        this.calculatedLayers = calculatedLayers;
    }
}
