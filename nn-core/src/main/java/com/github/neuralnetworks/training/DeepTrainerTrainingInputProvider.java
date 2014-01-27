package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.DNN;

/**
 * Training Input Provider for deep network trainers
 */
public class DeepTrainerTrainingInputProvider implements TrainingInputProvider {

    private TrainingInputProvider inputProvider;
    private DNN<?> dnn;
    private NeuralNetwork currentNN;
    private Set<Layer> calculatedLayers;
    private Map<Layer, Matrix> layerResults;

    public DeepTrainerTrainingInputProvider(TrainingInputProvider inputProvider, DNN<?> dnn, NeuralNetwork currentNN) {
	super();
	this.inputProvider = inputProvider;
	this.dnn = dnn;
	this.currentNN = currentNN;
	this.calculatedLayers = new HashSet<>();
	this.layerResults = new HashMap<>();
    }

    @Override
    public TrainingInputData getNextInput() {
	TrainingInputData input = inputProvider.getNextInput();

	if (input != null && dnn.getFirstNeuralNetwork() != currentNN) {
	    layerResults.put(dnn.getInputLayer(), input.getInput());
	    calculatedLayers.clear();
	    calculatedLayers.add(dnn.getInputLayer());
	    dnn.getLayerCalculator().calculate(dnn, currentNN.getInputLayer(), calculatedLayers, layerResults);
	    input = new TrainingInputDataImpl(layerResults.get(currentNN.getInputLayer()), input.getTarget());
	}

	return input;
    }

    @Override
    public int getInputSize() {
	return inputProvider.getInputSize();
    }

    @Override
    public void reset() {
	inputProvider.reset();
    }

    public TrainingInputProvider getInputProvider() {
        return inputProvider;
    }

    public void setInputProvider(TrainingInputProvider inputProvider) {
        this.inputProvider = inputProvider;
    }

    public DNN<?> getDnn() {
        return dnn;
    }

    public void setDnn(DNN<?> dnn) {
        this.dnn = dnn;
    }

    public NeuralNetwork getCurrentNN() {
        return currentNN;
    }

    public void setCurrentNN(NeuralNetwork currentNN) {
        this.currentNN = currentNN;
    }
}
