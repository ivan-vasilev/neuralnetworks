package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.DNN;
import com.github.neuralnetworks.calculation.LayerCalculator;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.training.events.MiniBatchFinishedEvent;
import com.github.neuralnetworks.training.events.TrainingFinishedEvent;
import com.github.neuralnetworks.training.events.TrainingStartedEvent;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * Default implementation for deep network trainer
 */
public class GreedyLayerDNNTrainer<N extends DNN<? extends NeuralNetwork>> extends Trainer<N> {

    public GreedyLayerDNNTrainer(Properties properties) {
	super(properties);
    }

    /* (non-Javadoc)
     * @see com.github.neuralnetworks.training.Trainer#train()
     * Child netwokrs are trained in sequential order. Each network has it's own Trainer.
     */
    @Override
    public void train() {
	triggerEvent(new TrainingStartedEvent(this));

	TrainingInputProvider inputProvider = getTrainingInputProvider();
	DNN<?> dnn = getNeuralNetwork();
	@SuppressWarnings("unchecked")
	DeepTrainingInputData deepInput = new DeepTrainingInputData((DNN<NeuralNetwork>) getNeuralNetwork(), getLayerCalculator());
	for (NeuralNetwork nn : dnn.getNeuralNetworks()) {
	    OneStepTrainer<?> trainer = getTrainers().get(nn);
	    inputProvider.reset();
	    deepInput.nn = nn;
	    while ((deepInput.baseInput = getTrainingInputProvider().getNextInput()) != null) {
		trainer.learnInput(deepInput);
		triggerEvent(new MiniBatchFromLayerFinished(this, deepInput.baseInput, trainer));
	    }

	    triggerEvent(new LayerTrainingFinished(this, trainer));
	}

	triggerEvent(new TrainingFinishedEvent(this));
    }

    public Map<NeuralNetwork, OneStepTrainer<?>> getTrainers() {
	return properties.getParameter(Constants.DEEP_TRAINERS);
    }

    /**
     * Wrapper object for the input data.
     * The "real" input is propagated through the deep network until the current child network is reached.
     * This result is returned from the wrapper as an input for the child network
     */
    private static class DeepTrainingInputData implements TrainingInputData {

	private DNN<NeuralNetwork> dnn;
	private NeuralNetwork nn;
	private TrainingInputData baseInput;
	private LayerCalculator calculator;
	private Map<Layer, Matrix> results;

	public DeepTrainingInputData(DNN<NeuralNetwork> dnn, LayerCalculator calculator) {
	    super();
	    this.dnn = dnn;
	    this.calculator = calculator;
	    this.results = new HashMap<>();
	}

	@Override
	public Matrix getInput() {
	    Matrix result = null;
	    Layer inputLayer = dnn.getInputLayer();
	    Layer currentLayer = dnn.getOutputLayer(nn);

	    if (inputLayer != currentLayer) {
		Matrix input = baseInput.getInput();
		Matrix output = results.get(currentLayer);
		if (output == null || output.getColumns() != input.getColumns()) {
		    output = new Matrix(currentLayer.getNeuronCount(), input.getColumns());
		    results.put(currentLayer, output);
		}
		
		results.put(inputLayer, input);

		Set<Layer> calculatedLayers = new HashSet<>();
		calculatedLayers.add(inputLayer);
		calculator.calculate(dnn, currentLayer, calculatedLayers, results);
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
     * Triggered when a training mini batch is finished
     */
    public static class MiniBatchFromLayerFinished extends MiniBatchFinishedEvent {

	private static final long serialVersionUID = 2155527437110587968L;

	public OneStepTrainer<?> currentTrainer;

	public MiniBatchFromLayerFinished(GreedyLayerDNNTrainer<?> source, TrainingInputData input, OneStepTrainer<?> currentTrainer) {
	    super(source, input);
	    this.currentTrainer = currentTrainer;
	}
    }

    /**
     * Triggered when a nested neural network has finished training
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
