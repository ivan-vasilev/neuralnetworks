package com.github.neuralnetworks.input.image;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.Serializable;

public class ImageResizeStrategy implements Serializable
{
	private static final long serialVersionUID = 1L;

	/**
	 * type of the resize
	 */
	public static enum ResizeType
	{
		SMALLEST_DIMENSION_RECT,
		SMALLEST_DIMENSION_SQUARE_RESIZE,
		SMALLEST_DIMENSION_SQUARE_CROP_MIDDLE,
	}

	protected ResizeType resizeType;
	protected int smallestDimension;

	/**
	 * set to true to resize in multiple steps for better quality
	 */
	protected boolean stepResize;

	public ImageResizeStrategy(ResizeType resizeType, int smallestDimension, boolean stepResize)
	{
		super();
		this.resizeType = resizeType;
		this.smallestDimension = smallestDimension;
		this.stepResize = stepResize;
	}

	public BufferedImage resize(BufferedImage input)
	{
		BufferedImage result = null;

		int targetWidth = 0, targetHeight = 0;
		switch (resizeType) {
		case SMALLEST_DIMENSION_RECT:
			if (input.getWidth() < input.getHeight())
			{
				targetWidth = smallestDimension;
				targetHeight = (int) (input.getHeight() / ((float) input.getWidth()) * smallestDimension);
			} else
			{
				targetWidth = (int) (input.getWidth() / ((float) input.getHeight()) * smallestDimension);
				targetHeight = smallestDimension;
			}
			break;
		case SMALLEST_DIMENSION_SQUARE_CROP_MIDDLE:
			if (input.getWidth() < input.getHeight())
			{
				input = input.getSubimage(0, (input.getHeight() - input.getWidth()) / 2, input.getWidth(), input.getWidth());
			} else
			{
				input = input.getSubimage((input.getWidth() - input.getHeight()) / 2, 0, input.getHeight(), input.getHeight());
			}
		case SMALLEST_DIMENSION_SQUARE_RESIZE:
			targetWidth = targetHeight = smallestDimension;
		}

		if (stepResize && targetWidth < input.getWidth() && targetHeight < input.getHeight())
		{
			int width = input.getWidth();
			int height = input.getHeight();
			while (width > targetWidth && height > targetHeight)
			{
				width = width / 2 < targetWidth ? targetWidth : width / 2;
				height = height / 2 < targetHeight ? targetHeight : height / 2;
				result = resize(input, width, height);
				input = result;
			}
		} else
		{
			result = resize(input, targetWidth, targetHeight); 
		}

		return result;
	}

	private BufferedImage resize(BufferedImage input, int targetWidth, int targetHeight)
	{
		BufferedImage result = new BufferedImage(targetWidth, targetHeight, input.getType());

		Graphics2D g = result.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
		g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g.drawImage(input, 0, 0, targetWidth, targetHeight, null);
		g.dispose();

		return result;
	}
}
