package com.github.neuralnetworks.input;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;

import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputProviderImpl;

/**
 * Input provider that takes float arrays from file
 */
public class SimpleFileInputProvider extends TrainingInputProviderImpl
{
	private static final long serialVersionUID = 1L;

	private String inputFile;
	private String targetFile;
	private transient RandomAccessFile input;
	private transient RandomAccessFile target;
	private transient FileBatchReader inputReader;
	private transient FileBatchReader targetReader;
	private int inputDimensions;
	private int targetDimensions;
	private int inputSize;

	public SimpleFileInputProvider(String inputFile, String targetFile, int inputDimensions, int targetDimensions, int inputSize)
	{
		this.inputFile = inputFile;
		this.targetFile = targetFile;
		this.inputDimensions = inputDimensions;
		this.targetDimensions = targetDimensions;
		this.inputSize = inputSize;

		initFiles();
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
			this.input = new RandomAccessFile(inputFile, "r");
			this.target = new RandomAccessFile(targetFile, "r");
		} catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}
	}

	@Override
	public int getInputSize()
	{
		return inputSize;
	}

	@Override
	public int getInputDimensions()
	{
		return inputDimensions;
	}

	@Override
	public int getTargetDimensions()
	{
		return targetDimensions;
	}

	@Override
	public void getNextInput(Tensor input)
	{
		if (inputReader == null)
		{
			inputReader = new FileBatchReader(this.input, (input.getElements().length - input.getStartIndex()) * 4);
		}

		ByteBuffer.wrap(inputReader.getNextInput()).asFloatBuffer().get(input.getElements());
	}

	@Override
	public void getNextTarget(Tensor target)
	{
		if (targetReader == null)
		{
			targetReader = new FileBatchReader(this.target, (target.getElements().length - target.getStartIndex()) * 4);
		}

		ByteBuffer.wrap(targetReader.getNextInput()).asFloatBuffer().get(target.getElements());
	}

	public void close()
	{
		try
		{
			if (target != null)
			{
				input.close();
			}

			if (target != null)
			{
				target.close();
			}
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	@Override
	public void afterBatch(TrainingInputData ti)
	{
		super.afterBatch(ti);
		if (currentInput == inputSize)
		{
			reset();
		}
	}

	@Override
	public void reset()
	{
		super.reset();

		try
		{
			inputReader = null;
			targetReader = null;

			if (input != null)
			{
				input.seek(0);
			}

			if (target != null)
			{
				target.seek(0);
			}
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}
}
