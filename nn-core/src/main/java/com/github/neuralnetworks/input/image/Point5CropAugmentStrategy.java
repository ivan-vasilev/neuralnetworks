package com.github.neuralnetworks.input.image;

import java.awt.image.BufferedImage;
import java.util.List;

/**
 * five crops per image (4 in the corners, 1 in the center)
 */
public class Point5CropAugmentStrategy implements ImageAugmentStrategy
{
	private static final long serialVersionUID = 1L;

	private int cropX;
	private int cropY;

	public Point5CropAugmentStrategy(int cropX, int cropY)
	{
		super();
		this.cropX = cropX;
		this.cropY = cropY;
	}

	@Override
	public void addAugmentedImages(List<BufferedImage> images)
	{
		int l = images.size();
		for (int i = 0; i < l; i++)
		{
			BufferedImage im = images.get(i);
			images.add(im.getSubimage(0, 0, cropX, cropY));
			images.add(im.getSubimage(im.getWidth() - cropX, 0, cropX, cropY));
			images.add(im.getSubimage(0, im.getHeight() - cropY, cropX, cropY));
			images.add(im.getSubimage(im.getWidth() - cropX, im.getHeight() - cropY, cropX, cropY));
			images.add(im.getSubimage((im.getWidth() - cropX) / 2, (im.getHeight() - cropY) / 2, cropX, cropY));
		}

		images.subList(0, l).clear();
	}

	@Override
	public int getSize()
	{
		return 5;
	}
}
