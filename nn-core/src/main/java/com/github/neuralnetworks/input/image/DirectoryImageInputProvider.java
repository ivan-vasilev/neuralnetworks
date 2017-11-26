package com.github.neuralnetworks.input.image;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.neuralnetworks.input.InputConverter;
import com.github.neuralnetworks.util.Pair;

/**
 * ImageInputProvider that retrieves all images from the subdirectories of the given directory. All images of one subdirectory will belong to one category.
 *
 */
public class DirectoryImageInputProvider extends ImageInputProvider
{
	private static final Logger logger = LoggerFactory.getLogger(DirectoryImageInputProvider.class);

	private static final long serialVersionUID = 1L;

	private File directory;
	private List<Pair<String, Integer>> filesAndCategory = new ArrayList<>();
	private List<BufferedImage> images;
	private Map<BufferedImage, Integer> mapCurrentImagesToIndex = Collections.synchronizedMap(new HashMap<>());;
	private int currentEl;
	private int inputDimensions = -1;
	private int targetDimensions;

	public DirectoryImageInputProvider(File directory)
	{
		this(null, directory);
	}

	public DirectoryImageInputProvider(InputConverter inputConverter, File directory)
	{
		super(inputConverter);
		this.directory = directory;

		if (!directory.exists())
		{
			throw new IllegalArgumentException("The directory " + directory + " doesn't exist!");
		}

		if (!directory.isDirectory())
		{
			throw new IllegalArgumentException("The directory " + directory + " isn't a directory!");
		}

		File[] listOfCategoryDirectories = directory.listFiles();

		if (listOfCategoryDirectories == null && listOfCategoryDirectories.length == 0)
		{
			throw new IllegalArgumentException("The directory " + directory + " doesn't contain subdirectories!");
		}

		int nrOfCategories = 0;
		int nrOfImages = 0;

		// list all files per category
		for (int i = 0; i < listOfCategoryDirectories.length; i++)
		{
			if (listOfCategoryDirectories[i].isDirectory())
			{
				int nrOfImagesOfThisCategory = 0;
				File[] listOfImages = listOfCategoryDirectories[i].listFiles();

				for (File image : listOfImages)
				{
					filesAndCategory.add(new Pair<String, Integer>(listOfCategoryDirectories[i].getName() + File.separatorChar + image.getName(), nrOfCategories));
					nrOfImages++;
					nrOfImagesOfThisCategory++;
				}

				if (nrOfImagesOfThisCategory > 0)
				{
					logger.info(nrOfImagesOfThisCategory + " images detected in category " + listOfCategoryDirectories[i].getName() + " (" + nrOfCategories + ")");
					nrOfCategories++;
				}
			}
		}

		targetDimensions = nrOfCategories;
		logger.info(nrOfImages + " images of " + nrOfCategories + " categories detected");

		// shuffle data
		Collections.shuffle(filesAndCategory);
	}

	@Override
	public int getInputSize()
	{
		return filesAndCategory.size() * super.getInputSize();
	}

	@Override
	protected float[] getNextTarget(BufferedImage image)
	{
		float[] target = new float[targetDimensions];
		target[filesAndCategory.get((int) mapCurrentImagesToIndex.remove(image)).getRight()] = 1;
		return target;
	}

	@Override
	public List<BufferedImage> getNextRawImages()
	{

		images = Collections.synchronizedList(new ArrayList<>(properties.getImagesBulkSize()));
		for (int i = 0; i < properties.getImagesBulkSize(); i++)
		{
			images.add(null);
		}


		IntStream stream = IntStream.range(0, properties.getImagesBulkSize());
		if (properties.getParallelPreprocessing())
		{
			stream = stream.parallel();
		}

		List<Integer> failedList = Collections.synchronizedList(new ArrayList<>());

		stream.forEach(i -> {
			int index = (currentEl + i) % filesAndCategory.size();
			Pair<String, Integer> imageCategoryPair = filesAndCategory.get(index);

			BufferedImage bufferedImage = null;
			try
			{
				bufferedImage = ImageIO.read(new File(directory, imageCategoryPair.getLeft()));
				mapCurrentImagesToIndex.put(bufferedImage, index);
			} catch (Exception e)
			{
				logger.warn("There is a problem with the image " + filesAndCategory.get((currentEl + i) % filesAndCategory.size()).getLeft(), e);
			}

			if (bufferedImage != null)
			{
				images.set(i, bufferedImage);
			} else
			{
				failedList.add(i);
			}

		});

		Collections.sort(failedList);
		for (int i = 0; i < failedList.size(); i++)
		{
			BufferedImage removed = images.remove((int) failedList.get(i));
			if (removed != null)
			{
				throw new IllegalStateException("An image that was null");
			}
		}

		currentEl += properties.getImagesBulkSize();

		return images;
	}

	@Override
	public int getInputDimensions()
	{
		if (inputDimensions == -1)
		{
			if (getProperties().getIsGrayscale())
			{
				inputDimensions = getProperties().getResizeStrategy().smallestDimension * getProperties().getResizeStrategy().smallestDimension;
			} else
			{
				inputDimensions = 3 * getProperties().getResizeStrategy().smallestDimension * getProperties().getResizeStrategy().smallestDimension;
			}

		}
		return inputDimensions;
	}

	@Override
	public int getTargetDimensions()
	{
		return targetDimensions;
	}

	@Override
	public String toString()
	{
		final StringBuilder sb = new StringBuilder("DirectoryImageInputProvider{");
		sb.append("targetDimensions=").append(targetDimensions);
		sb.append(", inputDimensions=").append(inputDimensions);
		sb.append(", currentEl=").append(currentEl);
		sb.append(", directory=").append(directory);
		sb.append('}');
		return sb.toString();
	}
}
