package com.github.neuralnetworks.testing;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.EventListener;
import java.util.List;

import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Util;

/**
 * 
 * implementations of this class
 * 
 */
public class Sampler {

    protected List<Trainer<?>> trainingConfigurations;
    protected List<EventListener> listeners;

    public Sampler() {
	super();
	this.trainingConfigurations = new ArrayList<>();
    }

    public Sampler(List<Trainer<?>> trainingConfigurations) {
	super();
	this.trainingConfigurations = trainingConfigurations;
    }

    public void sample() {
	Path newFile = Paths.get("results/" + new SimpleDateFormat("dd-MM-yyyyy HH-mm").format(new Date()) + ".txt");
	try (BufferedWriter writer = Files.newBufferedWriter(newFile, Charset.defaultCharset())) {
	    for (Trainer<?> t : trainingConfigurations) {
		long start = System.currentTimeMillis();
		t.train();
		long trainingTime = System.currentTimeMillis();
		writer.append("Training time: " + (trainingTime - start) / 1000 + "s");
		writer.newLine();
		t.test();
		long testingTime = System.currentTimeMillis();
		writer.append("Testing time: " + (testingTime - trainingTime) / 1000 + "s");
		writer.newLine();
		writer.append(Util.propertiesToString(t.getProperties()));
		writer.newLine();
		writer.append("========================================================================================================");
		writer.newLine();

		if (Environment.getInstance().isDebug()) {
		    System.out.println("Output error: " + t.getOutputError().getTotalNetworkError());
		    System.out.println("Training time: " + (trainingTime - start) / 1000 + "s");
		    System.out.println("Testing time: " + (testingTime - trainingTime) / 1000 + "s");
		}
	    }
	} catch (IOException exception) {
	    System.out.println("Error writing to file");
	}
    }

    public void addEventListener(EventListener listener) {
	if (listeners == null) {
	    listeners = new ArrayList<>();
	}

	listeners.add(listener);
    }

    public void removeEventListener(EventListener listener) {
	if (listeners != null) {
	    listeners.remove(listener);
	}
    }
    
    public List<Trainer<?>> getTrainingConfigurations() {
	return trainingConfigurations;
    }

    public void setTrainingConfigurations(List<Trainer<?>> trainingConfigurations) {
	this.trainingConfigurations = trainingConfigurations;
    }
}
