package com.github.neuralnetworks.input.image;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Random;

/**
 * Input provider for random images
 */
public class RandomImageInputProvider extends SimpleImageInputProvider
{
	private static final long serialVersionUID = 1L;

	public RandomImageInputProvider(int inputSize, int inputWidth, int inputHeight, int inputDimensions, int targetDimensions)
	{
		this(inputSize, inputWidth, inputHeight, inputDimensions, targetDimensions, new Random());
	}

	public RandomImageInputProvider(int inputSize, int inputWidth, int inputHeight, int inputDimensions, int targetDimensions, Random rand)
	{
		super(new ArrayList<>(), new ArrayList<>(), inputDimensions);

		// generate images
		for (int i = 0; i < inputSize; i++)
		{
			BufferedImage im = new BufferedImage(inputWidth, inputHeight, BufferedImage.TYPE_3BYTE_BGR);
			for (int j = 0; j < inputWidth; j++)
			{
				for (int k = 0; k < inputHeight; k++)
				{
					int b = rand.nextInt(256);
					int g = rand.nextInt(256);
					int r = rand.nextInt(256);
					im.setRGB(j, k, (b << 16) | (g << 8) | r);
				}
			}

			images.add(im);
		}

		// generate targets
		for (int i = 0; i < inputSize; i++)
		{
			float[] nextTarget = new float[targetDimensions];
			for (int j = 0; j < targetDimensions; j++)
			{
				nextTarget[j] = rand.nextFloat();
			}

			targets.add(nextTarget);
		}
	}
}
