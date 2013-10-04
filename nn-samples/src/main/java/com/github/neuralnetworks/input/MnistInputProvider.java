package com.github.neuralnetworks.input;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.github.neuralnetworks.training.TrainingInputData;
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
	private int currentIndex;
	private List<Integer> elementsOrder;
	private Random random;

	public MnistInputProvider(String imagesFile, String labelsFile) {
		super();
		try {
			this.images = new RandomAccessFile(imagesFile, "r");
			this.labels = new RandomAccessFile(labelsFile, "r");

			// magic numbers
			labels.readInt();
			labels.readInt();// read size
			images.readInt();
			inputSize = images.readInt();
			rows = images.readInt();
			cols = images.readInt();

			random = new Random();
			currentIndex = 0;
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
			currentIndex = elementsOrder.remove(random.nextInt(elementsOrder.size()));
			result = new TrainingInputData(getImage(currentIndex), getLabel(currentIndex));
		}

		return result;
	}

	@Override
	public int getInputSize() {
		return inputSize;
	}

	private float[] getImage(int index) {
		int size = cols * rows;
		float[] result = new float[size];
		try {
			images.seek(16 + size * index);
			for (int i = 0; i < size; i++) {
				result[i] = images.readUnsignedByte();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return result;
	}

	private int getLabel(int index) {
		Integer result = null;
		try {
			labels.seek(16 + index);
			result = labels.readUnsignedByte();
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
