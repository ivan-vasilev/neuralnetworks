package com.github.neuralnetworks.samples.mnist;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputProviderImpl;
import com.github.neuralnetworks.util.Util;

/**
 * MNIST data set with random order
 * Requires location of the MNIST images files (not included in the library)
 */
public class MnistInputProvider extends TrainingInputProviderImpl
{

	private static final long serialVersionUID = 1L;

	private String imagesFile;
	private String labelsFile;
	private transient RandomAccessFile images;
	private transient RandomAccessFile labels;
	private int rows;
	private int cols;
	private int inputSize;
	private List<Integer> elementsOrder;
	private int currentEl;
	private Random random;
	private byte[] current;

	public MnistInputProvider(String imagesFile, String labelsFile)
	{
		super();

		this.imagesFile = imagesFile;
		this.labelsFile = labelsFile;

		initFiles();

		try
		{
			// magic numbers
			images.readInt();
			inputSize = images.readInt();
			rows = images.readInt();
			cols = images.readInt();
			current = new byte[rows * cols];

			random = new Random();
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException
	{
		stream.defaultReadObject();
		initFiles();
	}

	private void initFiles()
	{
		try
		{
			this.images = new RandomAccessFile(imagesFile, "r");
			this.labels = new RandomAccessFile(labelsFile, "r");
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	@Override
	public void beforeBatch(TrainingInputData ti)
	{
		try
		{
			Tensor input = ti.getInput();
			Tensor target = ti.getTarget();

			if (target != null)
			{
				Util.fillArray(ti.getTarget().getElements(), 0);
			}

			int mb = input.getDimensions()[0];
			for (int i = 0; i < mb; i++)
			{
				if (elementsOrder.size() == 0)
				{
					resetOrder();
				}

				currentEl = elementsOrder.remove(random.nextInt(elementsOrder.size()));

				// images
				images.seek(16 + cols * rows * currentEl);
				images.readFully(current);
				for (int j = 0; j < cols * rows; j++)
				{
					input.getElements()[input.getStartIndex() + i * getInputDimensions() + j] = current[j] & 0xFF;
				}

				// target
				if (target != null)
				{
					labels.seek(8 + currentEl);
					target.getElements()[target.getStartIndex() + i * 10 + labels.readUnsignedByte()] = 1;
				}
			}
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	@Override
	public void getNextInput(Tensor input)
	{
	}

	@Override
	public void getNextTarget(Tensor target)
	{
	}

	@Override
	public void reset()
	{
		super.reset();
		resetOrder();
	}

	public void resetOrder()
	{
		elementsOrder = new ArrayList<Integer>(inputSize);
		for (int i = 0; i < inputSize; i++)
		{
			elementsOrder.add(i);
		}
	}

	public byte[] getCurrent()
	{
		return current;
	}

	@Override
	public int getInputSize()
	{
		return inputSize;
	}

	@Override
	public int getTargetDimensions()
	{
		return 10;
	}

	@Override
	public int getInputDimensions()
	{
		return rows * cols;
	}

	public int getRows()
	{
		return rows;
	}

	public int getCols()
	{
		return cols;
	}
}
