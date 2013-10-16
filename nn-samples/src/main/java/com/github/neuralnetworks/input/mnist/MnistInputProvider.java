package com.github.neuralnetworks.input.mnist;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.github.neuralnetworks.input.InputConverter;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputDataImpl;
import com.github.neuralnetworks.training.TrainingInputProvider;

/**
 * 
 * MNIST data set with random order
 * 
 */
public class MnistInputProvider implements TrainingInputProvider {

    private RandomAccessFile images;
    private RandomAccessFile labels;
    private int rows;
    private int cols;
    private int inputSize;
    private final int batchSize;
    private List<Integer> elementsOrder;
    private Random random;
    private final InputConverter inputConverter;
    private final InputConverter targetConverter;
    private Integer[][] tempImages;

    public MnistInputProvider(String imagesFile, String labelsFile, int batchSize, InputConverter inputConverter, InputConverter targetConverter) {
	super();

	this.batchSize = batchSize;
	this.inputConverter = inputConverter;
	this.targetConverter = targetConverter;

	try {
	    this.images = new RandomAccessFile(imagesFile, "r");
	    this.labels = new RandomAccessFile(labelsFile, "r");

	    // magic numbers
	    images.readInt();
	    inputSize = images.readInt();
	    rows = images.readInt();
	    cols = images.readInt();

	    random = new Random();
	    elementsOrder = new ArrayList<Integer>(inputSize);
	    for (int i = 0; i < inputSize; i++) {
		elementsOrder.add(i);
	    }
	} catch (IOException e) {
	    e.printStackTrace();
	}
    }

    @Override
    public TrainingInputData getNextInput() {
	TrainingInputData result = null;
	if (elementsOrder.size() > 0) {
	    int length = elementsOrder.size() > batchSize ? batchSize : elementsOrder.size();
	    int[] indexes = new int[length];
	    for (int i = 0; i < length; i++) {
		indexes[i] = elementsOrder.remove(random.nextInt(elementsOrder.size()));
	    }

	    result = new TrainingInputDataImpl(getImages(indexes), getLabels(indexes), inputConverter, targetConverter);
	}

	return result;
    }

    @Override
    public int getInputSize() {
	return inputSize;
    }

    private Integer[][] getImages(int[] indexes) {
	int size = cols * rows;
	if (tempImages == null || tempImages.length != indexes.length) {
	    tempImages = new Integer[indexes.length][size];
	}

	try {
	    for (int i = 0; i < indexes.length; i++) {
		images.seek(16 + size * indexes[i]);
		for (int j = 0; j < size; j++) {
		    tempImages[i][j] = images.readUnsignedByte();
		}
	    }
	} catch (IOException e) {
	    e.printStackTrace();
	}

	return tempImages;
    }

    private Integer[] getLabels(int indexes[]) {
	Integer[] result = new Integer[indexes.length];

	try {
	    for (int i = 0; i < indexes.length; i++) {
		labels.seek(8 + indexes[i]);
		result[i] = labels.readUnsignedByte();
	    }
	} catch (IOException e) {
	    e.printStackTrace();
	}

	return result;
    }

    public int getRows() {
	return rows;
    }

    public int getCols() {
	return cols;
    }
}
