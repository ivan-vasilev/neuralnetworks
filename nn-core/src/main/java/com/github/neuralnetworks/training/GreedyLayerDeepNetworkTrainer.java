package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.DeepNeuralNetwork;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.LayerCalculator;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * 
 * Default implementation for deep network trainer
 *
 */
public class GreedyLayerDeepNetworkTrainer extends Trainer<DeepNeuralNetwork> {

    public GreedyLayerDeepNetworkTrainer(Properties properties) {
	super(properties);
    }

    @Override
    public void train() {
	TrainingInputProvider inputProvider = getTrainingInputProvider();
	DeepNeuralNetwork dnn = getNeuralNetwork();
	DeepTrainingInputData deepInput = new DeepTrainingInputData(getNeuralNetwork(), getLayerCalculator());
	for (NeuralNetwork nn : dnn.getNeuralNetworks()) {
	    OneStepTrainer<?> trainer = getTrainers().get(nn);
	    inputProvider.reset();
	    deepInput.nn = nn;
	    while ((deepInput.baseInput = getTrainingInputProvider().getNextInput()) != null) {
		trainer.learnInput(deepInput);
		triggerEvent(new SampleFinishedEvent(this, deepInput.baseInput));
	    }

	    triggerEvent(new TrainingFinishedEvent(this));
	}
    }

    public Map<NeuralNetwork, OneStepTrainer<?>> getTrainers() {
	return properties.getParameter(Constants.DEEP_TRAINERS);
    }

    private static class DeepTrainingInputData implements TrainingInputData {

	private DeepNeuralNetwork dnn;
	private NeuralNetwork nn;
	private TrainingInputData baseInput;
	private LayerCalculator calculator;
	private Map<Layer, Matrix> results;

	public DeepTrainingInputData(DeepNeuralNetwork dnn, LayerCalculator calculator) {
	    super();
	    this.dnn = dnn;
	    this.calculator = calculator;
	    this.results = new HashMap<>();
	}

	@Override
	public Matrix getInput() {
	    Matrix result = null;
	    Layer inputLayer = dnn.getInputLayer();
	    Layer currentLayer = nn.getInputLayer();

	    if (inputLayer != currentLayer) {
		Matrix input = baseInput.getInput();
		Matrix output = results.get(currentLayer);
		if (output == null || output.getColumns() != input.getColumns()) {
		    output = new Matrix(currentLayer.getNeuronCount(), input.getColumns());
		    results.put(currentLayer, output);
		}
		
		results.put(inputLayer, baseInput.getInput());

		Set<Layer> calculatedLayers = new HashSet<>();
		calculatedLayers.add(inputLayer);
		calculator.calculate(calculatedLayers, results, currentLayer);
		result = results.get(currentLayer);
	    } else {
		result = baseInput.getInput();
	    }

	    return result;
	}

	@Override
	public Matrix getTarget() {
	    return baseInput.getTarget();
	}
    }
}
