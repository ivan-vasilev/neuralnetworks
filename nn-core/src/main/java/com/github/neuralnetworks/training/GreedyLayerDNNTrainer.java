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
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * Default implementation for deep network trainer
 */
public class GreedyLayerDNNTrainer extends Trainer<DeepNeuralNetwork> {

    public GreedyLayerDNNTrainer(Properties properties) {
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
		triggerEvent(new SampleFromLayerFinished(this, deepInput.baseInput, trainer));
	    }

	    triggerEvent(new LayerTrainingFinished(this, trainer));
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

    /**
     * this event is triggered when a training sample is finished
     */
    public static class SampleFromLayerFinished extends SampleFinishedEvent {

	private static final long serialVersionUID = 2155527437110587968L;

	public OneStepTrainer<?> currentTrainer;

	public SampleFromLayerFinished(GreedyLayerDNNTrainer source, TrainingInputData input, OneStepTrainer<?> currentTrainer) {
	    super(source, input);
	    this.currentTrainer = currentTrainer;
	}
    }

    /**
     * this event is triggered when a nested neural network has finished training
     */
    public static class LayerTrainingFinished extends TrainingEvent {

	private static final long serialVersionUID = 2155527437110587968L;

	public OneStepTrainer<?> currentTrainer;

	public LayerTrainingFinished(Trainer<?> source, OneStepTrainer<?> currentTrainer) {
	    super(source);
	    this.currentTrainer = currentTrainer;
	}
    }
}
