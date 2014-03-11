package com.github.neuralnetworks.training.events;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.training.Trainer;

/**
 * Time/error log
 */
public class LogTrainingListener implements TrainingEventListener {

    private static final long serialVersionUID = 1L;

    private String name;
    private long startTime;
    private long finishTime;
    private long miniBatchTime;
    private long miniBatchTotalTime;
    private long lastMiniBatchFinishTime;
    private int miniBatches;

    /**
     * log minibatches time
     */
    private boolean logMiniBatches;

    /**
     * dispaly input/target/networkOutput for each testing example
     */
    private boolean logTestResults;

    private boolean isTesting;

    public LogTrainingListener(String name) {
	super();
	this.name = name;
    }

    public LogTrainingListener(String name, boolean logTestResults, boolean logMiniBatches) {
	super();
	this.name = name;
	this.logTestResults = logTestResults;
	this.logMiniBatches = logMiniBatches;
    }

    @Override
    public void handleEvent(TrainingEvent event) {
	if (event instanceof TrainingStartedEvent || event instanceof TestingStartedEvent) {
	    reset();
	    lastMiniBatchFinishTime = startTime = System.currentTimeMillis();

	    if (event instanceof TrainingStartedEvent) {
		isTesting = false;
		System.out.println("TRAINING " + name + "...");
	    } else if (event instanceof TestingStartedEvent) {
		isTesting = true;
		System.out.println();
		System.out.println("TESTING " + name + "...");
	    }
	} else if (event instanceof TrainingFinishedEvent || event instanceof TestingFinishedEvent) {
	    finishTime = System.currentTimeMillis();
	    String s = System.getProperty("line.separator");

	    StringBuilder sb = new StringBuilder();
	    sb.append(((finishTime - startTime) / 1000f) + " s  total time" + s);
	    sb.append((miniBatchTotalTime / (miniBatches * 1000f)) + " s  per minibatch of " + miniBatches + " mini batches" + s);
	    if (event instanceof TestingFinishedEvent) {
		Trainer<?> t = (Trainer<?>) event.getSource();
		OutputError oe = t.getOutputError();
		sb.append(oe.getTotalErrorSamples() + "/" + oe.getTotalInputSize() + " samples (" + oe.getTotalNetworkError() + ", " + (oe.getTotalNetworkError() * 100) + "%) error" + s + s);
	    }

	    System.out.print(sb.toString());
	} else if (event instanceof MiniBatchFinishedEvent) {
	    miniBatches++;
	    miniBatchTime += System.currentTimeMillis() - lastMiniBatchFinishTime;
	    miniBatchTotalTime += System.currentTimeMillis() - lastMiniBatchFinishTime;
	    lastMiniBatchFinishTime = System.currentTimeMillis();

	    StringBuilder sb = new StringBuilder();
	    String s = System.getProperty("line.separator");

	    if (miniBatchTime / 5000 > 0 && (logMiniBatches || (isTesting && logTestResults))) {
		sb.append(miniBatches + " minibatches in " + (miniBatchTotalTime / 1000f) + " s" + s);
		miniBatchTime = 0;
	    }

	    // log test results
	    if (isTesting && logTestResults) {
		MiniBatchFinishedEvent mbe = (MiniBatchFinishedEvent) event;
		if (mbe.getResults() != null) {
		    Matrix input = mbe.getData().getInput();
		    Matrix target = mbe.getData().getTarget();
		    Trainer<?> t = (Trainer<?>) mbe.getSource();
		    Matrix networkOutput = mbe.getResults().getValues(t.getNeuralNetwork().getOutputLayer());

		    for (int i = 0; i < input.getColumns(); i++) {
			sb.append(s);
			sb.append("Input:  ");
			for (int j = 0; j < input.getRows(); j++) {
			    sb.append(input.get(j, i)).append("  ");
			}

			sb.append(s);
			sb.append("Output: ");
			for (int j = 0; j < networkOutput.getRows(); j++) {
			    sb.append(networkOutput.get(j, i)).append("  ");
			}

			sb.append(s);
			sb.append("Target: ");
			for (int j = 0; j < target.getRows(); j++) {
			    sb.append(target.get(j, i)).append("  ");
			}
		    }
		    sb.append(s).append(s);
		}
	    }

	    System.out.print(sb.toString());
	} else if (event instanceof TestingFinishedEvent) {
	    miniBatches++;
	}
    }

    private void reset() {
	startTime = finishTime = miniBatchTotalTime = lastMiniBatchFinishTime = miniBatches = 0;
    }
}
