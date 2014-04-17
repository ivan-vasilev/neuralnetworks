package com.github.neuralnetworks.events;

import java.util.EventObject;

import com.github.neuralnetworks.training.Trainer;

/**
 * Base class for events during the training phase
 */
public class TrainingEvent extends EventObject {

    private static final long serialVersionUID = 1171094415041968041L;

    public TrainingEvent(Trainer<?> source) {
	super(source);
    }
}
