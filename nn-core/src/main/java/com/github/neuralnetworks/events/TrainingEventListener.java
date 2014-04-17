package com.github.neuralnetworks.events;

import java.io.Serializable;
import java.util.EventListener;

/**
 * Base listener for training events
 */
public interface TrainingEventListener extends EventListener, Serializable {
    public void handleEvent(TrainingEvent event);
}
