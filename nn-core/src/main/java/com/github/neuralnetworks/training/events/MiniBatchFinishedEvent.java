package com.github.neuralnetworks.training.events;

import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainingInputData;

/**
 * Event, triggered when a single batch finishes training
 */
public class MiniBatchFinishedEvent extends TrainingEvent {

    private static final long serialVersionUID = -5239379347414855784L;

    private TrainingInputData data;
    private ValuesProvider results;

    public MiniBatchFinishedEvent(Trainer<?> source, TrainingInputData data, ValuesProvider results) {
	super(source);
	this.data = data;
	this.results = results;
    }

    public TrainingInputData getData() {
        return data;
    }

    public ValuesProvider getResults() {
        return results;
    }
}
