package com.github.neuralnetworks.performance.tests;

import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.training.events.MiniBatchFinishedEvent;
import com.github.neuralnetworks.training.events.MiniBatchStartedEvent;
import com.github.neuralnetworks.training.events.NewInputEvent;
import com.github.neuralnetworks.training.events.TestingFinishedEvent;
import com.github.neuralnetworks.training.events.TestingStartedEvent;
import com.github.neuralnetworks.training.events.TrainingFinishedEvent;
import com.github.neuralnetworks.training.events.TrainingStartedEvent;

/**
 * Created by chass on 03.11.14.
 */
public class PerformanceEventListener implements TrainingEventListener {

	private static final long serialVersionUID = -3480701244688117371L;

	private long startTime = -1L;
    private long totalTime = -1L;

    @Override
    public void handleEvent(TrainingEvent event) {


        if (event instanceof TrainingStartedEvent || event instanceof TestingStartedEvent) {
            this.startTime = System.currentTimeMillis();
            this.totalTime = -1;

        } else if (event instanceof TrainingFinishedEvent || event instanceof TestingFinishedEvent) {
            if(this.startTime > -1L){
                this.totalTime = (System.currentTimeMillis()-this.startTime);
            }

        } else if (event instanceof NewInputEvent) {

        } else if (event instanceof MiniBatchStartedEvent) {

        } else if (event instanceof MiniBatchFinishedEvent) {

        } else if (event instanceof TestingFinishedEvent) {

        }

    }

    public long getTrainingRunTimeMs(){
        return this.totalTime;
    }
}
