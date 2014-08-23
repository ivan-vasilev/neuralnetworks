package com.github.neuralnetworks.input;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import com.github.neuralnetworks.training.TrainingInputProviderImpl;

/**
 * Input provider for CSV files. Values should be real numbers only. The separator is comma. Target file is not required.
 * !!! Important - the target values are provided to the network as they are read from the file. This means, that when you have actual target values 1, 2, 3 it is your responsibility to convert them into 1,0,0; 0,1,0 and 0,0,1, if your network requires this.
 */
public class CSVInputProvider extends TrainingInputProviderImpl {

    private static final long serialVersionUID = 5067933748794269003L;

    private String inputFile;
    private String targetFile;
    private BufferedReader inputReader;
    private BufferedReader targetReader;
    private float[] nextInput;
    private float[] nextTarget;
    private int inputSize;

    public CSVInputProvider(String inputFile, String targetFile) {
	super();
	this.inputFile = inputFile;
	this.targetFile = targetFile;

	inputSize = (int) getInputReader().lines().count();
    }

    public CSVInputProvider(InputConverter targetConverter, String inputFile, String targetFile) {
	super(targetConverter);
	this.inputFile = inputFile;
	this.targetFile = targetFile;

	inputSize = (int) getInputReader().lines().count();
    }

    @Override
    public int getInputSize() {
	return inputSize;
    }

    @Override
    public float[] getNextInput() {
	return nextInput;
    }

    @Override
    public float[] getNextTarget() {
	return nextTarget;
    }

    public String getInputFile() {
	return inputFile;
    }

    public String getTargetFile() {
	return targetFile;
    }

    @Override
    public void beforeSample() {
	try {
	    // input
	    BufferedReader ir = getInputReader();
	    String line = ir.readLine();
	    if (line == null) {
		inputReader.close();
		inputReader = null;
		ir = getInputReader();
		line = ir.readLine();
	    }

	    String[] split = line.split(",");
	    if (nextInput == null) {
		nextInput = new float[split.length];
	    }

	    for (int i = 0; i < nextInput.length; i++) {
		nextInput[i] = Float.parseFloat(split[i]);
	    }

	    // target
	    if (targetFile != null) {
		BufferedReader tr = getTargetReader();
		line = tr.readLine();
		if (line == null) {
		    targetReader.close();
		    targetReader = null;
		    tr = getTargetReader();
		    line = tr.readLine();
		}
		
		split = line.split(",");
		if (nextTarget == null) {
		    nextTarget = new float[split.length];
		}

		for (int i = 0; i < nextTarget.length; i++) {
		    nextTarget[i] = Float.parseFloat(split[i]);
		}
	    }
	} catch (IOException e) {
	    e.printStackTrace();
	}
    }

    @Override
    public void reset() {
	super.reset();

	if (inputReader != null) {
	    try {
		inputReader.close();
		inputReader = null;
	    } catch (IOException e) {
		e.printStackTrace();
	    }
	}

	if (targetReader != null) {
	    try {
		targetReader.close();
		targetReader = null;
	    } catch (IOException e) {
		e.printStackTrace();
	    }
	}
    }

    private BufferedReader getInputReader() {
	if (inputReader == null) {
	    try {
		inputReader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(inputFile))));
	    } catch (IOException e) {
		e.printStackTrace();
	    }
	}

	return inputReader;
    }
    
    private BufferedReader getTargetReader() {
	if (targetReader == null && targetFile != null) {
	    try {
		targetReader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(targetFile))));
	    } catch (IOException e) {
		e.printStackTrace();
	    }
	}

	return targetReader;
    }
}
