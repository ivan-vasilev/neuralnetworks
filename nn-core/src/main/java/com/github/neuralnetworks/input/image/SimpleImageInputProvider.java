package com.github.neuralnetworks.input.image;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Simple image input provider
 */
public class SimpleImageInputProvider extends ImageInputProvider
{
	private static final long serialVersionUID = 1L;

	protected List<BufferedImage> images;
	protected List<float[]> targets;
	private int inputDimensions;

	public SimpleImageInputProvider(List<BufferedImage> images, List<float[]> targets, int inputDimensions)
	{
		super();
		this.images = images;
		this.targets = targets;
		this.inputDimensions = inputDimensions;
	}

	@Override
	public int getInputSize()
	{
		return images.size() * super.getInputSize();
	}

	@Override
	public int getInputDimensions()
	{
		return inputDimensions;
	}

	@Override
	public int getTargetDimensions()
	{
		return targets.get(0).length;
	}

	public List<BufferedImage> getImages()
	{
		return images;
	}

	public List<float[]> getTargets()
	{
		return targets;
	}

	@Override
	public List<BufferedImage> getNextRawImages()
	{
		List<BufferedImage> result = new ArrayList<>();
		for (int i = currentInput; i < properties.getImagesBulkSize() + currentInput; i++)
		{
			result.add(images.get(i % images.size()));
		}

		return result;
	}

	@Override
	protected float[] getNextTarget(BufferedImage image)
	{
		float[] result = null;
		if (targets != null)
		{
			result = targets.get(images.indexOf(image));
		}

		return result;
	}
}
