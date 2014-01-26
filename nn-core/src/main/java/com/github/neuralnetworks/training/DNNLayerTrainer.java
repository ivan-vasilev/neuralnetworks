package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.DNN;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.training.events.MiniBatchFinishedEvent;
import com.github.neuralnetworks.training.events.TrainingFinishedEvent;
import com.github.neuralnetworks.training.events.TrainingStartedEvent;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * Default implementation for deep network trainer
 */
public class DNNLayerTrainer extends Trainer<DNN<? extends NeuralNetwork>> {

    public DNNLayerTrainer(Properties properties) {
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

	Map<Layer, Matrix> layerResults = new HashMap<>();
	Set<Layer> calculatedLayers = new HashSet<>();
	TrainingInputData input = null;

	for (NeuralNetwork nn : dnn.getNeuralNetworks()) {
	    OneStepTrainer<?> trainer = getTrainers().get(nn);
	    inputProvider.reset();

	    while ((input = getTrainingInputProvider().getNextInput()) != null) {
		TrainingInputData trainerInput = null;
		if (dnn.getFirstNeuralNetwork() != nn) {
		    layerResults.put(dnn.getInputLayer(), input.getInput());
		    calculatedLayers.clear();
		    calculatedLayers.add(dnn.getInputLayer());
		    dnn.getLayerCalculator().calculate(dnn, nn.getInputLayer(), calculatedLayers, layerResults);
		    trainerInput = new TrainingInputDataImpl(layerResults.get(nn.getInputLayer()), input.getTarget());
		} else {
		    trainerInput = input;
		}

		trainer.learnInput(trainerInput);

		triggerEvent(new MiniBatchFromLayerFinished(this, input, trainer));
	    }

	    triggerEvent(new LayerTrainingFinished(this, trainer));
	}

	triggerEvent(new TrainingFinishedEvent(this));
    }

    public Map<NeuralNetwork, OneStepTrainer<?>> getTrainers() {
	return properties.getParameter(Constants.LAYER_TRAINERS);
    }

    /**
     * Triggered when a training mini batch is finished
     */
    public static class MiniBatchFromLayerFinished extends MiniBatchFinishedEvent {

	private static final long serialVersionUID = 2155527437110587968L;

	public OneStepTrainer<?> currentTrainer;

	public MiniBatchFromLayerFinished(DNNLayerTrainer source, TrainingInputData input, OneStepTrainer<?> currentTrainer) {
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
