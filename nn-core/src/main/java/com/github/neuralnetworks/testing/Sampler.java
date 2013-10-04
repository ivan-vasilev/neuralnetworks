package com.github.neuralnetworks.testing;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.util.Util;

/**
 *
 * implementations of this class
 *
 */
public class Sampler {

	protected List<Trainer<?>> trainingConfigurations;

	public Sampler() {
		super();
		this.trainingConfigurations = new ArrayList<>();
	}

	public Sampler(List<Trainer<?>> trainingConfigurations) {
		super();
		this.trainingConfigurations = trainingConfigurations;
	}

	public void sample() {
		Path newFile = Paths.get(DateFormat.getDateInstance().format(new Date()) + ".txt");
		try (BufferedWriter writer = Files.newBufferedWriter(newFile, Charset.defaultCharset())) {
			for (Trainer<?> t : trainingConfigurations) {
				t.train();
				writer.append(Util.propertiesToString(t.getProperties()));
				writer.newLine();
				writer.append("========================================================================================================");
				writer.newLine();
			}
		} catch (IOException exception) {
			System.out.println("Error writing to file");
		}
	}

	public List<Trainer<?>> getTrainingConfigurations() {
		return trainingConfigurations;
	}

	public void setTrainingConfigurations(List<Trainer<?>> trainingConfigurations) {
		this.trainingConfigurations = trainingConfigurations;
	}
}
