package com.github.neuralnetworks.samples.mnist;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.github.neuralnetworks.input.InputConverter;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputDataImpl;
import com.github.neuralnetworks.training.TrainingInputProviderImpl;
import com.github.neuralnetworks.util.Matrix;

/**
 * MNIST data set with random order
 */
public class MnistInputProvider extends TrainingInputProviderImpl {

    private static final long serialVersionUID = 1L;

    private RandomAccessFile images;
    private RandomAccessFile labels;
    private int epochs;
    private int currentEpoch;
    private int rows;
    private int cols;
    private int inputSize;
    private final int batchSize;
    private List<Integer> elementsOrder;
    private Random random;
    private final InputConverter targetConverter;
    private Matrix tempImages;
    private byte[] current;

    public MnistInputProvider(String imagesFile, String labelsFile, int batchSize, int epochs, InputConverter targetConverter) {
	super();

	this.epochs = epochs;
	this.batchSize = batchSize;
	this.targetConverter = targetConverter;

	try {
	    this.images = new RandomAccessFile(imagesFile, "r");
	    this.labels = new RandomAccessFile(labelsFile, "r");

	    // magic numbers
	    images.readInt();
	    inputSize = images.readInt();
	    rows = images.readInt();
	    cols = images.readInt();
	    current = new byte[rows * cols];

	    random = new Random();
	} catch (IOException e) {
	    e.printStackTrace();
	}
    }

    @Override
    public TrainingInputData getNextUnmodifiedInput() {
	TrainingInputData result = null;

	if (elementsOrder.size() == 0 && currentEpoch < epochs) {
	    resetOrder();
	    currentEpoch++;
	}

	if (elementsOrder.size() > 0) {
	    int length = elementsOrder.size() > batchSize ? batchSize : elementsOrder.size();
	    int[] indexes = new int[length];
	    for (int i = 0; i < length; i++) {
		indexes[i] = elementsOrder.remove(random.nextInt(elementsOrder.size()));
	    }

	    Matrix input = getImages(indexes);

	    Matrix target = targetConverter.convert(getLabels(indexes));

	    result = new TrainingInputDataImpl(input, target);
	}

	return result;
    }

    @Override
    public void reset() {
	currentEpoch = 1;
	resetOrder();
    }

    public void resetOrder() {
	elementsOrder = new ArrayList<Integer>(inputSize);
	for (int i = 0; i < inputSize; i++) {
	    elementsOrder.add(i);
	}
    }

    @Override
    public int getInputSize() {
	return inputSize * epochs;
    }

    private Matrix getImages(int[] indexes) {
	int size = cols * rows;
	if (tempImages == null || tempImages.getRows() != indexes.length) {
	    tempImages = new Matrix(size, indexes.length);
	}

	try {
	    for (int i = 0; i < indexes.length; i++) {
		images.seek(16 + size * indexes[i]);
		images.readFully(current);
		for (int j = 0; j < size; j++) {
		    tempImages.set(current[j] & 0xFF, j, i);
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
