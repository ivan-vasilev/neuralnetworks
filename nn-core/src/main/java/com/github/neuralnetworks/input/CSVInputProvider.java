package com.github.neuralnetworks.input;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.training.TrainingInputProviderImpl;

/**
 * Input provider for CSV files. Values should be real numbers only. The separator is comma. Target file is not required.
 * !!! Important - the target values are provided to the network as they are read from the file. This means, that when you have actual target values 1, 2, 3 it is your responsibility to convert them
 * into 1,0,0; 0,1,0 and 0,0,1, if your network requires this.
 */
public class CSVInputProvider extends TrainingInputProviderImpl
{

	private static final long serialVersionUID = 5067933748794269003L;

	private File inputFile;
	private File targetFile;
	private BufferedReader inputReader;
	private BufferedReader targetReader;
	private int inputSize;
	private int inputDimensions;
	private int targetDimensions;

	public CSVInputProvider(File inputFile, File targetFile)
	{
		super();
		this.inputFile = inputFile;
		this.targetFile = targetFile;

		inputSize = (int) getInputReader().lines().count();
		inputDimensions = getInputReader().lines().findFirst().get().split(",").length;

		if (getTargetReader() != null) 
		{
			targetDimensions = getTargetReader().lines().findFirst().get().split(",").length;
		}
	}

	public CSVInputProvider(InputConverter targetConverter, File inputFile, File targetFile)
	{
		super(targetConverter);
		this.inputFile = inputFile;
		this.targetFile = targetFile;

		inputSize = (int) getInputReader().lines().count();
		inputDimensions = getInputReader().lines().findFirst().get().split(",").length;
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

	@SuppressWarnings("resource")
	@Override
	public void getNextInput(Tensor input)
	{
		try
		{
			// input
			BufferedReader ir = getInputReader();
			int mb = input.getDimensions()[0];
			float[] elements = input.getElements();

			for (int i = 0; i < mb; i++)
			{
				String line = ir.readLine();
				if (line == null)
				{
					inputReader.close();
					inputReader = null;
					ir = getInputReader();
					line = ir.readLine();
				}

				String[] split = line.split(",");

				for (int j = 0; j < elements.length; j++)
				{
					elements[input.getStartIndex() + i * getInputDimensions() + j] = Float.parseFloat(split[j]);
				}
			}
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	@SuppressWarnings("resource")
	@Override
	public void getNextTarget(Tensor target)
	{
		if (targetReader != null && target != null)
		{
			try
			{
				// input
				BufferedReader ir = getTargetReader();
				int mb = target.getDimensions()[0];
				float[] elements = target.getElements();

				for (int i = 0; i < mb; i++)
				{
					String line = ir.readLine();
					if (line == null)
					{
						targetReader.close();
						targetReader = null;
						ir = getTargetReader();
						line = ir.readLine();
					}

					String[] split = line.split(",");

					for (int j = 0; j < elements.length; j++)
					{
						elements[target.getStartIndex() + i * split.length + j] = Float.parseFloat(split[j]);
					}
				}
			} catch (IOException e)
			{
				e.printStackTrace();
			}
		}
	}

	@Override
	public void reset()
	{
		super.reset();

		if (inputReader != null)
		{
			try
			{
				inputReader.close();
				inputReader = null;
			} catch (IOException e)
			{
				e.printStackTrace();
			}
		}

		if (targetReader != null)
		{
			try
			{
				targetReader.close();
				targetReader = null;
			} catch (IOException e)
			{
				e.printStackTrace();
			}
		}
	}

	private BufferedReader getInputReader()
	{
		if (inputReader == null)
		{
			try
			{
				inputReader = new BufferedReader(new InputStreamReader(new FileInputStream(inputFile)));
			} catch (IOException e)
			{
				e.printStackTrace();
			}
		}

		return inputReader;
	}

	private BufferedReader getTargetReader()
	{
		if (targetReader == null && targetFile != null)
		{
			try
			{
				targetReader = new BufferedReader(new InputStreamReader(new FileInputStream(targetFile)));
			} catch (IOException e)
			{
				e.printStackTrace();
			}
		}

		return targetReader;
	}
}
