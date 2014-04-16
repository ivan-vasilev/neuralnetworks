package com.github.neuralnetworks.training;

import java.util.HashSet;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.DNN;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.TensorFactory;

/**
 * Training Input Provider for deep network trainers
 */
public class DeepTrainerTrainingInputProvider extends TrainingInputProviderImpl {

    private static final long serialVersionUID = 1L;

    private TrainingInputProvider inputProvider;
    private TrainingInputData inputDataBase;
    private DNN<?> dnn;
    private NeuralNetwork currentNN;
    private Set<Layer> calculatedLayers;
    private ValuesProvider layerResults;

    public DeepTrainerTrainingInputProvider(TrainingInputProvider inputProvider, DNN<?> dnn, NeuralNetwork currentNN, int batchSize) {
	super();
	this.inputProvider = inputProvider;
	this.dnn = dnn;
	this.currentNN = currentNN;
	this.calculatedLayers = new HashSet<>();
	this.layerResults = TensorFactory.tensorProvider(batchSize, Environment.getInstance().getUseSharedMemory(), dnn, currentNN);
	this.inputDataBase = new TrainingInputDataImpl(layerResults.get(dnn.getInputLayer()));
    }

    @Override
    public void after(TrainingInputData ti) {
	if (dnn.getFirstNeuralNetwork() != currentNN) {
	    inputProvider.populateNext(inputDataBase);
	    calculatedLayers.clear();
	    calculatedLayers.add(dnn.getInputLayer());
	    dnn.getLayerCalculator().calculate(dnn, currentNN.getInputLayer(), calculatedLayers, layerResults);
	    TensorFactory.copy(layerResults.get(currentNN.getInputLayer()), ti.getInput());
	}
    }

    @Override
    public float[] getNextInput() {
	return inputProvider.getNextInput();
    }

    @Override
    public float[] getNextTarget() {
	return inputProvider.getNextTarget();
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
