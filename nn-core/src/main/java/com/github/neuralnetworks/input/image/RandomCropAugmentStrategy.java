package com.github.neuralnetworks.input.image;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Random;

/**
 * random crops
 */
public class RandomCropAugmentStrategy implements ImageAugmentStrategy
{
	private static final long serialVersionUID = 1L;

	private int cropX;
	private int cropY;
	private int crops;
	private Random random;

	public RandomCropAugmentStrategy(int cropX, int cropY, int crops)
	{
		super();
		this.cropX = cropX;
		this.cropY = cropY;
		this.crops = crops;
		this.random = new Random();
	}

	@Override
	public void addAugmentedImages(List<BufferedImage> images)
	{
		int l = images.size();
		for (int i = 0; i < l; i++)
		{
			BufferedImage im = images.get(i);
			for (int j = 0; j < crops; j++)
			{
				images.add(im.getSubimage(random.nextInt(im.getWidth() - cropX), random.nextInt(im.getHeight() - cropY), cropX, cropY));
			}
		}

		images.subList(0, l).clear();
	}

	@Override
	public int getSize()
	{
		return 1;
	}
}
