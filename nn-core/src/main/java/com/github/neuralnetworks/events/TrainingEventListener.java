package com.github.neuralnetworks.events;

import java.util.EventListener;

public interface TrainingEventListener extends EventListener {
    public void handleEvent(TrainingEvent event);
}
