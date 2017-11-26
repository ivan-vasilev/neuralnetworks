package com.github.neuralnetworks.input.image;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Composite augmentation strategy (it applies different strategies to the images
 */
public class CompositeAugmentStrategy implements ImageAugmentStrategy
{
	private static final long serialVersionUID = 1L;

	private List<ImageAugmentStrategy> strategies;

	public CompositeAugmentStrategy(ImageAugmentStrategy... strategy)
	{
		super();
		this.strategies = new ArrayList<>();
		for (ImageAugmentStrategy s : strategy)
		{
			strategies.add(s);
		}
	}

	@Override
	public void addAugmentedImages(List<BufferedImage> images)
	{
		for (ImageAugmentStrategy strategy : strategies)
		{
			strategy.addAugmentedImages(images);
		}
	}

	public void addStrategy(ImageAugmentStrategy strategy)
	{
		strategies.add(strategy);
	}

	@Override
	public int getSize()
	{
		int result = 1;
		for (ImageAugmentStrategy s : strategies)
		{
			result *= s.getSize();
		}

		return result;
	}
}
