package com.github.neuralnetworks.input;

import java.util.Random;

/**
 * random input provider (based on simple input provider)
 */
public class RandomInputProvider extends SimpleInputProvider
{
	private static final long serialVersionUID = 1L;

	public RandomInputProvider(int inputSize, int inputDimensions)
	{
		this(inputSize, inputDimensions, new Random());
	}

	public RandomInputProvider(int inputSize, int inputDimensions, Random rand)
	{
		super(new float[inputSize][inputDimensions]);
		for (float[] f : getInput())
		{
			for (int i = 0; i < f.length; i++)
			{
				f[i] = rand.nextFloat() * 2 - 1;
			}
		}
	}

	public RandomInputProvider(int inputSize, int inputDimensions, int targetDimensions)
	{
		this(inputSize, inputDimensions, targetDimensions, new Random());
	}

	public RandomInputProvider(int size, int inputDimensions, int targetDimensions, Random rand)
	{
		super(new float[size][inputDimensions], new float[size][targetDimensions]);
		for (float[] f : getInput())
		{
			for (int i = 0; i < f.length; i++)
			{
				f[i] = rand.nextFloat() * 2 - 1;
			}
		}

		for (float[] f : getTarget())
		{
			for (int i = 0; i < f.length; i++)
			{
				f[i] = rand.nextFloat() * 2 - 1;
			}
		}
	}
}
