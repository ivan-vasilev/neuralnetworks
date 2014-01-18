package com.github.neuralnetworks.training.events;

import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.training.Trainer;

/**
 * Time/error log
 */
public class LogTrainingListener implements TrainingEventListener {

    private long startTime;
    private long finishTime;
    private long miniBatchTotalTime;
    private long lastMiniBatchFinishTime;
    private int miniBatches;
    private boolean logMiniBatches = false;

    public LogTrainingListener() {
	super();
    }

    public LogTrainingListener(boolean logMiniBatches) {
	super();
	this.logMiniBatches = logMiniBatches;
    }

    @Override
    public void handleEvent(TrainingEvent event) {
	if (event instanceof TrainingStartedEvent || event instanceof TestingStartedEvent) {
	    reset();
	    lastMiniBatchFinishTime = startTime = System.currentTimeMillis();

	    if (event instanceof TrainingStartedEvent) {
		System.out.println("TRAINING:");
	    } else if (event instanceof TestingStartedEvent) {
		System.out.println();
		System.out.println("TESTING:");
	    }
	} else if (event instanceof TrainingFinishedEvent || event instanceof TestingFinishedEvent) {
	    finishTime = System.currentTimeMillis();
	    String s = System.getProperty("line.separator");

	    StringBuilder sb = new StringBuilder();
	    sb.append(((finishTime - startTime) / 1000f) + " s  total time" + s);
	    sb.append((miniBatchTotalTime / (miniBatches * 1000f)) + " s  per minibatch of " + miniBatches + " mini batches" + s);
	    if (event instanceof TestingFinishedEvent) {
		Trainer<?> t = (Trainer<?>) event.getSource();
		sb.append(t.getOutputError().getTotalNetworkError() + " (" + (t.getOutputError().getTotalNetworkError() * 100) + "%) error" + s);
	    }

	    System.out.print(sb.toString());
	} else if (event instanceof MiniBatchFinishedEvent) {
	    miniBatches++;
	    long miniBatchTime = System.currentTimeMillis() - lastMiniBatchFinishTime;
	    miniBatchTotalTime += miniBatchTime;
	    lastMiniBatchFinishTime = System.currentTimeMillis();

	    if (logMiniBatches) {
		System.out.println("MB " + miniBatches + " " + (miniBatchTime / 1000f) + " s");
	    }
	} else if (event instanceof TestingFinishedEvent) {
	    miniBatches++;
	}
    }

    private void reset() {
	startTime = finishTime = miniBatchTotalTime = lastMiniBatchFinishTime = miniBatches = 0;
    }
}
