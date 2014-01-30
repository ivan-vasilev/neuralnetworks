package com.github.neuralnetworks.training.events;

import java.util.Map;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainingInputData;

/**
 * Event, triggered when a single batch finishes training
 */
public class MiniBatchFinishedEvent extends TrainingEvent {

    private static final long serialVersionUID = -5239379347414855784L;

    private TrainingInputData data;
    private Map<Layer, Matrix> results;

    public MiniBatchFinishedEvent(Trainer<?> source, TrainingInputData data, Map<Layer, Matrix> results) {
	super(source);
	this.data = data;
	this.results = results;
    }

    public TrainingInputData getData() {
        return data;
    }

    public Map<Layer, Matrix> getResults() {
        return results;
    }
}
