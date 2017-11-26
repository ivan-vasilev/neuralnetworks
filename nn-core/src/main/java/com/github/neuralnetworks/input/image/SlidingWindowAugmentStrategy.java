package com.github.neuralnetworks.input.image;

import java.awt.image.BufferedImage;
import java.util.List;

/**
 * Augmentation strategy that provides sliding window over an image
 */
public class SlidingWindowAugmentStrategy implements ImageAugmentStrategy
{
	private static final long serialVersionUID = 1L;

	protected int width;
	protected int height;
	protected int strideX;
	protected int strideY;
	protected int size;

	public SlidingWindowAugmentStrategy(int width, int height, int strideX, int strideY)
	{
		super();
		this.width = width;
		this.height = height;
		this.strideX = strideX;
		this.strideY = strideY;
	}

	public SlidingWindowAugmentStrategy(int width, int height)
	{
		this(width, height, 1, 1);
	}

	@Override
	public void addAugmentedImages(List<BufferedImage> images)
	{
		int l = images.size();

		for (int i = 0; i < l; i++)
		{
			BufferedImage source = images.get(i);

			for (int x = 0; x <= source.getWidth() - width; x += strideX)
			{
				for (int y = 0; y <= source.getHeight() - height; y += strideY)
				{
					images.add(source.getSubimage(x, y, width, height));
				}
			}
		}

		images.subList(0, l).clear();
	}

	@Override
	public int getSize()
	{
		return size;
	}

	public void setSize(int size)
	{
		this.size = size;
	}

	public int getWidth()
	{
		return width;
	}

	public void setWidth(int width)
	{
		this.width = width;
	}

	public int getHeight()
	{
		return height;
	}

	public void setHeight(int height)
	{
		this.height = height;
	}

	public int getStrideX()
	{
		return strideX;
	}

	public void setStrideX(int strideX)
	{
		this.strideX = strideX;
	}

	public int getStrideY()
	{
		return strideY;
	}

	public void setStrideY(int strideY)
	{
		this.strideY = strideY;
	}
}
