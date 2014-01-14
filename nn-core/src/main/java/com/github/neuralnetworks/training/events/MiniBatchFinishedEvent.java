package com.github.neuralnetworks.training.events;

import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainingInputData;

/**
 * Event, triggered when a single batch finishes training
 */
public class MiniBatchFinishedEvent extends TrainingEvent {

    private static final long serialVersionUID = -5239379347414855784L;

    public TrainingInputData data;

    public MiniBatchFinishedEvent(Trainer<?> source, TrainingInputData data) {
	super(source);
	this.data = data;
    }
}
