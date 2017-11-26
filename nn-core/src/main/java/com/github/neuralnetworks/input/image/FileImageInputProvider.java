package com.github.neuralnetworks.input.image;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

import javax.imageio.ImageIO;

import com.github.neuralnetworks.input.InputConverter;

/**
 * ImageInputProvider that retrieves all images from a list of files (full paths required)
 */
public abstract class FileImageInputProvider extends ImageInputProvider
{
	private static final long serialVersionUID = 1L;

	private transient BufferedImage[] images;
	private int currentEl;
	private Random random;
	private List<Integer> elementsOrder;

	/**
	 * map between filenames and read images;
	 */
	protected transient Map<BufferedImage, String> imageToFile;

	public FileImageInputProvider(InputConverter inputConverter)
	{
		this(inputConverter, null);
	}

	public FileImageInputProvider(InputConverter inputConverter, Random random)
	{
		super(inputConverter);
		this.random = random;
	}

	@Override
	protected void init()
	{
		super.init();
		this.imageToFile = Collections.synchronizedMap(new HashMap<>());
	}

	protected abstract List<String> getFiles();

	@Override
	public int getInputSize()
	{
		return getFiles().size() * super.getInputSize();
	}

	@Override
	public List<BufferedImage> getNextRawImages()
	{
		if (images == null)
		{
			images = new BufferedImage[properties.getImagesBulkSize()];
		}

		IntStream stream = IntStream.range(0, properties.getImagesBulkSize());
		if (properties.getParallelPreprocessing())
		{
			stream = stream.parallel();
		}

		if (random != null && elementsOrder == null)
		{
			elementsOrder = Collections.synchronizedList(new ArrayList<>(getFiles().size()));
		}

		stream.forEach(i -> {
			try
			{
				int id = 0;
				if (random != null)
				{
					synchronized (elementsOrder)
					{
						if (elementsOrder.size() == 0)
						{
							for (int j = 0; j < getFiles().size(); j++)
							{
								elementsOrder.add(j);
							}
						}

						id = elementsOrder.remove(random.nextInt(elementsOrder.size()));
					}
				} else
				{
					id = currentEl % getFiles().size() + i;
				}

				images[i] = ImageIO.read(new File(getFiles().get(id)));

				imageToFile.put(images[i], getFiles().get(id));
			} catch (Exception e)
			{
				e.printStackTrace();
			}
		});

		currentEl += properties.getImagesBulkSize();

		return Arrays.asList(images);
	}

	@Override
	public void reset()
	{
		super.reset();
		imageToFile.clear();
	}

	/**
	 * clear the map
	 */
	@Override
	protected void freeArray(float[] array)
	{
		synchronized (lock)
		{
			BufferedImage im = arrayToAugmented.get(array);
			BufferedImage rawImage = null;
			if (im != null)
			{
				rawImage = augmentedToRaw.get(im);
			}

			super.freeArray(array);

			if (!augmentedToRaw.values().contains(rawImage))
			{
				imageToFile.remove(rawImage);
			}
		}
	}

	@Override
	public abstract int getInputDimensions();

	@Override
	public abstract int getTargetDimensions();
}
