package com.github.neuralnetworks.training.events;

import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputDataImpl;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.TensorFactory;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Listener for early stopping of the training
 */
public class EarlyStoppingListener implements TrainingEventListener {

    private static final long serialVersionUID = 1L;

    /**
     * input provider for the cross-validation data
     */
    private TrainingInputProvider inputProvider;

    /**
     * how much minibatches before each training
     */
    private int validationFrequency;

    /**
     * What error is considered acceptable to stop
     */
    private float acceptanceError;

    private boolean isTraining;

    public EarlyStoppingListener(TrainingInputProvider inputProvider, int validationFrequency, float acceptanceError) {
	super();
	this.validationFrequency = validationFrequency;
	this.inputProvider = inputProvider;
	this.acceptanceError = acceptanceError;
    }

    @Override
    public void handleEvent(TrainingEvent event) {
	if (event instanceof TrainingStartedEvent) {
	    isTraining = true;
	} else if (event instanceof TrainingFinishedEvent) {
	    isTraining = false;
	} else if (event instanceof MiniBatchFinishedEvent && isTraining) {
	    MiniBatchFinishedEvent mbe = (MiniBatchFinishedEvent) event;
	    if (mbe.getBatchCount() % validationFrequency == 0) {
		OneStepTrainer<?> t = (OneStepTrainer<?>) event.getSource();
		NeuralNetwork n = t.getNeuralNetwork();

		if (n.getLayerCalculator() != null) {
		    OutputError outputError = t.getOutputError();
		    outputError.reset();
		    inputProvider.reset();

		    ValuesProvider vp = mbe.getResults();
		    if (vp == null) {
			vp = TensorFactory.tensorProvider(n, 1, Environment.getInstance().getUseDataSharedMemory());
		    }
		    if (vp.get(outputError) == null) {
			vp.add(outputError, vp.get(n.getInputLayer()).getDimensions());
		    }
		    TrainingInputData input = new TrainingInputDataImpl(vp.get(n.getInputLayer()), vp.get(outputError));

		    Set<Layer> calculatedLayers = new UniqueList<>();
		    for (int i = 0; i < inputProvider.getInputSize(); i ++) {
			inputProvider.populateNext(input);
			calculatedLayers.clear();
			calculatedLayers.add(n.getInputLayer());

			n.getLayerCalculator().calculate(n, n.getOutputLayer(), calculatedLayers, vp);

			outputError.addItem(vp.get(n.getOutputLayer()), input.getTarget());
		    }

		    float e = outputError.getTotalNetworkError();
		    if (e <= acceptanceError) {
			System.out.println("Stopping at error " + e + " (" + (e * 100) + "%) for " + mbe.getBatchCount() + " minibatches");
			t.stopTraining();
		    }
		}
	    }
	}
    }
}
