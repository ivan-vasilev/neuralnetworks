package com.github.neuralnetworks.input.image;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import com.github.neuralnetworks.input.InputConverter;

/**
 * ImageInputProvider that retrieves all images from the original source image based on the sliding window technique
 */
public abstract class SlidingWindowImageInputProvider extends ImageInputProvider
{
	private static final long serialVersionUID = 1L;

	private List<BufferedImage> sourceImages;
	private BufferedImage[] images;
	private int width;
	private int height;
	private int strideX;
	private int strideY;
	private int currentImage;
	private int currentX;
	private int currentY;
	private int inputSize;

	public SlidingWindowImageInputProvider(InputConverter inputConverter, List<BufferedImage> sourceImages, int width, int height, int strideX, int strideY)
	{
		super(inputConverter);
		this.sourceImages = sourceImages;
		this.width = width;
		this.height = height;
		this.strideX = strideX;
		this.strideY = strideY;
	}

	@Override
	public int getInputSize()
	{
		if (inputSize == 0)
		{
			for (BufferedImage si : sourceImages)
			{
				inputSize += (si.getWidth() - width) / strideX + (si.getHeight() - height) / strideY;
			}
		}

		return inputSize * super.getInputSize();
	}

	@Override
	public List<BufferedImage> getNextRawImages()
	{
		if (images == null)
		{
			images = new BufferedImage[properties.getImagesBulkSize()];
		}

		IntStream stream = IntStream.range(0, properties.getImagesBulkSize());

		stream.forEach(i -> {
			if (sourceImages.get(currentImage).getWidth() - width > currentX)
			{
				currentX++;
			} else if (sourceImages.get(currentImage).getHeight() - height > currentY)
			{
				currentY++;
				currentX = 0;
			} else
			{
				currentImage = (currentImage + 1) % sourceImages.size();
				currentY = currentX = 0;
			}

			images[i] = sourceImages.get(currentImage).getSubimage(currentX, currentY, width, height);
		});

		return Arrays.asList(images);
	}

	@Override
	public int getInputDimensions()
	{
		return width * height * 3;
	}

	@Override
	public int getTargetDimensions()
	{
		return 0;
	}
}
