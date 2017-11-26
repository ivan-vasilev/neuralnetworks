package com.github.neuralnetworks.events;

import java.io.Serializable;
import java.util.EventListener;

/**
 * Event listener for propagation events
 */
public interface PropagationEventListener extends EventListener, Serializable
{
	public void handleEvent(PropagationEvent event);
}
