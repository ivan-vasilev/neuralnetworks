package com.github.neuralnetworks.events;

import java.util.EventListener;

/**
 * Event listener for propagation events
 */
public interface PropagationEventListener extends EventListener {
    public void handleEvent(PropagationEvent event);
}
