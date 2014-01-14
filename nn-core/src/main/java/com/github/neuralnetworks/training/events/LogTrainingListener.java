package com.github.neuralnetworks.training.events;

import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.training.Trainer;

/**
 * Time/error log
 */
public class LogTrainingListener implements TrainingEventListener {

    private final static Logger LOGGER = Logger.getLogger("training");
    static {
	LOGGER.setLevel(Level.ALL);
	ConsoleHandler handler = new ConsoleHandler();
	handler.setFormatter(new SimpleFormatter());
	LOGGER.addHandler(handler);
    }

    private long startTime;
    private long finishTime;
    private long miniBatchTotalTime;
    private long lastMiniBatchFinishTime;
    private int miniBatches;

    @Override
    public void handleEvent(TrainingEvent event) {
	if (event instanceof TrainingStartedEvent || event instanceof TestingStartedEvent) {
	    reset();
	    lastMiniBatchFinishTime = startTime = System.currentTimeMillis();
	} else if (event instanceof TrainingFinishedEvent || event instanceof TestingFinishedEvent) {
	    finishTime = System.currentTimeMillis();
	    StringBuilder sb = new StringBuilder();
	    String s = System.getProperty("line.separator");
	    sb.append(s);

	    if (event instanceof TrainingFinishedEvent) {
		sb.append("TRAINING:" + s);
	    } else if (event instanceof TestingFinishedEvent) {
		sb.append("TESTING:" + s);
	    }

	    sb.append(((finishTime - startTime) / 1000) + "s (" + (finishTime - startTime) + "ms) total time" + s);
	    sb.append((miniBatchTotalTime / miniBatches) + " ms (" + (miniBatchTotalTime / (miniBatches * 1000)) + " s) per minibatch of " + miniBatches + " mini batches" + s);
	    if (event instanceof TestingFinishedEvent) {
		Trainer<?> t = (Trainer<?>) event.getSource();
		sb.append(t.getOutputError().getTotalNetworkError() + " (" + (t.getOutputError().getTotalNetworkError() * 100) + "%) error" + s);
	    }

	    LOGGER.info(sb.toString());
	} else if (event instanceof MiniBatchFinishedEvent) {
	    miniBatchTotalTime += System.currentTimeMillis() - lastMiniBatchFinishTime;
	    lastMiniBatchFinishTime = System.currentTimeMillis();
	    miniBatches++;
	} else if (event instanceof TestingFinishedEvent) {
	    miniBatches++;
	}
    }

    private void reset() {
	startTime = finishTime = miniBatchTotalTime = lastMiniBatchFinishTime = miniBatches = 0;
    }
}
