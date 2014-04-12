package com.github.neuralnetworks.training;

import java.util.HashSet;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.DNN;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.util.Environment;

/**
 * Training Input Provider for deep network trainers
 */
public class DeepTrainerTrainingInputProvider implements TrainingInputProvider {

    private static final long serialVersionUID = 1L;

    private TrainingInputProvider inputProvider;
    private DNN<?> dnn;
    private NeuralNetwork currentNN;
    private Set<Layer> calculatedLayers;
    private ValuesProvider layerResults;

    public DeepTrainerTrainingInputProvider(TrainingInputProvider inputProvider, DNN<?> dnn, NeuralNetwork currentNN) {
	super();
	this.inputProvider = inputProvider;
	this.dnn = dnn;
	this.currentNN = currentNN;
	this.calculatedLayers = new HashSet<>();
	this.layerResults = Environment.getInstance().getValuesProvider(dnn);
    }

    @Override
    public TrainingInputData getNextInput() {
	TrainingInputData input = inputProvider.getNextInput();

	if (input != null && dnn.getFirstNeuralNetwork() != currentNN) {
	    layerResults.replace(dnn.getInputLayer(), input.getInput());
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

    public DNN<?> getDnn() {
        return dnn;
    }

    public NeuralNetwork getCurrentNN() {
        return currentNN;
    }
}
