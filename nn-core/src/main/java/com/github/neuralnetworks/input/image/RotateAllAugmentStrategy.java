package com.github.neuralnetworks.input.image;

import java.awt.image.BufferedImage;
import java.util.List;

/**
 * Flips images by 90/180/270 degrees
 */
public class RotateAllAugmentStrategy implements ImageAugmentStrategy
{
	private static final long serialVersionUID = 1L;

	@Override
	public void addAugmentedImages(List<BufferedImage> images)
	{
		for (int i = 0, l = images.size(); i < l; i++)
		{
			BufferedImage source = images.get(i);

			// 90 rotation
			BufferedImage im90 = new BufferedImage(source.getWidth(), source.getHeight(), source.getType());
			for (int row = 0; row < source.getHeight(); row++)
			{
				for (int col = 0; col < source.getWidth(); col++)
				{
					im90.setRGB(source.getWidth() - row - 1, col, source.getRGB(col, row));
				}
			}

			images.add(im90);

			// 180 rotation
			BufferedImage im180 = new BufferedImage(source.getWidth(), source.getHeight(), source.getType());
			for (int row = 0; row < source.getHeight(); row++)
			{
				for (int col = 0; col < source.getWidth(); col++)
				{
					im180.setRGB(source.getWidth() - row - 1, col, im90.getRGB(col, row));
				}
			}

			images.add(im180);

			// 270 rotation
			BufferedImage im270 = new BufferedImage(source.getWidth(), source.getHeight(), source.getType());
			for (int row = 0; row < source.getHeight(); row++)
			{
				for (int col = 0; col < source.getWidth(); col++)
				{
					im270.setRGB(source.getWidth() - row - 1, col, im180.getRGB(col, row));
				}
			}

			images.add(im270);
		}
	}

	@Override
	public int getSize()
	{
		return 4;
	}
}
