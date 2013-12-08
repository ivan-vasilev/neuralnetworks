package com.github.neuralnetworks.events;

import java.util.EventListener;

/**
 * Base listener for training events
 */
public interface TrainingEventListener extends EventListener {
    public void handleEvent(TrainingEvent event);
}
