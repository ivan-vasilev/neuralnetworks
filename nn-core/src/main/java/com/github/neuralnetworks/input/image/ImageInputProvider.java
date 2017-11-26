package com.github.neuralnetworks.input.image;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.neuralnetworks.input.InputConverter;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputProviderImpl;
import com.github.neuralnetworks.util.Properties;

/**
 * Base class for image input providers. Supported image augmentation operations: all affine transforms and cropping. Specify maxRotationAngle explicitly! <br>
 * <br>
 * comments:<br>
 * - the augmentedImagesBufferSize variable must be greater than 0 (default 1) and larger than bulk size (if not then the program blocks itself) <br>
 * - the imagesBulkSize variable must be set and greater than 0 (default null) but smaller than augmentedImagesBufferSize
 */
public abstract class ImageInputProvider extends TrainingInputProviderImpl
{

	private static final Logger logger = LoggerFactory.getLogger(ImageInputProvider.class);

	private static final long serialVersionUID = 1L;

	protected ImageInputProviderProperties properties;

	/**
	 * augmented images to raw images
	 */
	public transient Map<BufferedImage, BufferedImage> augmentedToRaw;

	/**
	 * augmented arrays to raw images
	 */
	public transient Map<float[], BufferedImage> arrayToAugmented;

	/**
	 * raw images order in the current minibatch
	 */
	public transient Map<Integer, BufferedImage> rawMBPosition;

	/**
	 * pool of arrays to be used for augmented images
	 */
	private transient Set<float[]> freeArrays;

	/**
	 * lock for synchronized access over various arrays
	 */
	protected transient Object lock;
	/**
	 * augmented images floats
	 */
	private transient BlockingQueue<float[]> values;

	/**
	 * thread for retrieving images
	 */
	private transient BackgroundThread thread;

	/**
	 * current arrays
	 */
	private transient List<float[]> currentArrays;

	public ImageInputProvider()
	{
		this(null);
	}

	public ImageInputProvider(InputConverter inputConverter)
	{
		super(inputConverter);
		this.properties = new ImageInputProviderProperties();

		init();
	}

	protected void init()
	{
		this.augmentedToRaw = Collections.synchronizedMap(new HashMap<>());
		this.arrayToAugmented = Collections.synchronizedMap(new HashMap<>());
		this.rawMBPosition = Collections.synchronizedMap(new HashMap<>());
		this.freeArrays = Collections.synchronizedSet(new HashSet<>());
		this.currentArrays = Collections.synchronizedList(new ArrayList<>());
		this.lock = new Object();
	}

	private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException
	{
		stream.defaultReadObject();
		init();
	}

	@Override
	public void getNextInput(Tensor input)
	{
		int mb = input.getDimensions()[0];

		currentArrays.clear();
		for (int i = 0; i < mb; i++)
		{
			try
			{
				if (getValues().size() == 0)
				{
					createArrays();
				}

				currentArrays.add(getValues().take());
			} catch (InterruptedException e)
			{
				e.printStackTrace();
			}
		}

		IntStream stream = IntStream.range(0, mb);
		if (properties.getParallelPreprocessing())
		{
			stream = stream.parallel();
		}

		stream.forEach(i -> {
			float[] a = currentArrays.get(i);
			rawMBPosition.put(i, augmentedToRaw.get(arrayToAugmented.get(a)));
			System.arraycopy(a, 0, input.getElements(), input.getStartIndex() + i * getInputDimensions(), a.length);
		});
	}

	@Override
	public void getNextTarget(Tensor target)
	{
		IntStream stream = IntStream.range(0, currentArrays.size());
		if (properties.getParallelPreprocessing())
		{
			stream = stream.parallel();
		}

		stream.forEach(i -> {
			float[] t = getNextTarget(augmentedToRaw.get(arrayToAugmented.get(currentArrays.get(i))));
			if (t != null)
			{
				System.arraycopy(t, 0, target.getElements(), target.getStartIndex() + i * getTargetDimensions(), t.length);
			}
		});
	}

	@SuppressWarnings("unused")
	protected float[] getNextTarget(BufferedImage image)
	{
		return null;
	}

	@Override
	public void beforeBatch(TrainingInputData ti)
	{
		super.beforeBatch(ti);
		for (float[] a : currentArrays)
		{
			freeArray(a);
		}
	}

	/**
	 * @return next image from the set
	 */
	public abstract List<BufferedImage> getNextRawImages();

	/**
	 * create float arrays based on the raw images
	 */
	public void createArrays()
	{
		if (properties.getParallelPreprocessing())
		{
			if (properties.getBackgroundThread())
			{
				if (thread == null)
				{
					thread = new BackgroundThread();
					thread.setRunning(true);
					thread.start();
				}
			} else
			{
				createAugmentedImages(getNextRawImages(), augmentedToRaw);
				createAugmentedArrays(augmentedToRaw, arrayToAugmented, values);
			}
		} else
		{
			createAugmentedImages(getNextRawImages(), augmentedToRaw);
			createAugmentedArrays(augmentedToRaw, arrayToAugmented, values);
		}
	}

	/**
	 * prepares augmented images based on raw images
	 */
	public void createAugmentedImages(List<BufferedImage> rawImages, Map<BufferedImage, BufferedImage> augmentedToRaw)
	{
		if (properties.getIsGrayscale() == null && rawImages.size() > 0)
		{
			Integer type = rawImages.get(0).getType();
			properties.setIsGrayscale(type == BufferedImage.TYPE_BYTE_GRAY || type == BufferedImage.TYPE_USHORT_GRAY);
		}

		Stream<BufferedImage> s = properties.getParallelPreprocessing() ? rawImages.parallelStream() : rawImages.stream();
		s.forEach(i -> {
			BufferedImage augmented = i;
			if (properties.getResizeStrategy() != null)
			{
				augmented = properties.getResizeStrategy().resize(i);
			}

			if (properties.getAugmentStrategy() != null)
			{
				List<BufferedImage> augmentedImages = new ArrayList<>();
				augmentedImages.add(augmented);
				properties.getAugmentStrategy().addAugmentedImages(augmentedImages);
				for (BufferedImage bi : augmentedImages)
				{
					augmentedToRaw.put(bi, i);
				}
			} else
			{
				augmentedToRaw.put(augmented, i);
			}
		});
	}

	/**
	 * prepares augmented arrays based on augmented images
	 */
	public void createAugmentedArrays(Map<BufferedImage, BufferedImage> augmentedToRaw, Map<float[], BufferedImage> arrayToAugmented, BlockingQueue<float[]> values)
	{
		List<BufferedImage> images = null;
		synchronized (lock)
		{
			images = augmentedToRaw.keySet().stream().filter(image -> !arrayToAugmented.values().contains(image)).collect(Collectors.toList());
		}
		Stream<BufferedImage> s = properties.getParallelPreprocessing() ? images.parallelStream() : images.stream();

		List<float[]> arrays = properties.getParallelPreprocessing() ? Collections.synchronizedList(new ArrayList<>()) : new ArrayList<>();
		s.forEach(image -> {
// for convenience			
//			try
//			{
//				File outputfile = new File(new Random().nextInt() + ".png");
//				ImageIO.write(image, "png", outputfile);
//			} catch (IOException e)
//			{
//			}

			float[] nextInput = imageToFloat(((DataBufferByte) image.getRaster().getDataBuffer()).getData(), image.getWidth() * image.getHeight());

			if (nextInput.length != getInputDimensions())
			{
				throw new IllegalStateException("Incorrect vector size: " + nextInput.length + " instead of " + getInputSize());
			}

			arrayToAugmented.put(nextInput, image);
			arrays.add(nextInput);
		});

		try
		{
			for (float[] a : arrays)
			{
				if (a != null)
				{
					values.put(a);
				}
			}
		} catch (Exception e)
		{
			e.printStackTrace();
		}
	}

	/**
	 * pixels are BGR ordered, this function will swap them to RGB format !!!!
	 * 
	 * @param pixels
	 * @param size
	 * @return
	 */
	protected float[] imageToFloat(byte[] pixels, int size)
	{
		int pixelDataLength = pixels.length / size;

		float[] nextInput = null;
		synchronized (lock)
		{
			if (!freeArrays.isEmpty())
			{
				nextInput = freeArrays.iterator().next();
				freeArrays.remove(nextInput);
			} else if (properties.getIsGrayscale())
			{
				nextInput = new float[size];
			} else
			{
				nextInput = new float[size * 3];
			}
		}

		float scaleColors = properties.getScaleColors() ? 255 : 1;
		boolean subtractMean = properties.getSubtractMean();
		float[] meanArray = properties.getSubtractArray();
		if (subtractMean && meanArray != null)
		{
			throw new RuntimeException("Using subtractMean and meanArray together is not allowed");
		}
		
		if (properties.getIsGrayscale())
		{
			if (subtractMean)
			{
				double mean = 0;
				for (int j = 0; j < size; j++)
				{
					nextInput[j] = pixels[j * pixelDataLength + pixelDataLength - 1] & 0xFF;
					mean += nextInput[j];
				}

				float m = (float) (mean / nextInput.length);
				for (int i = 0; i < size; i++)
				{
					nextInput[i] = (nextInput[i] - m) / scaleColors;
				}
			} else if (meanArray != null)
			{
				for (int j = 0; j < size; j++)
				{
					nextInput[j] = (pixels[j * pixelDataLength + pixelDataLength - 1] & 0xFF) / scaleColors - meanArray[j];
				}
			} else
			{
				for (int j = 0; j < size; j++)
				{
					nextInput[j] = (pixels[j * pixelDataLength + pixelDataLength - 1] & 0xFF) / scaleColors;
				}
			}
		} else
		{
			if (pixelDataLength == 1)
			{
				logger.warn("Transform black/white to rgb image!");
				// create 3 channels images
				byte[] rgbPixels = new byte[pixels.length * 3];

				for (int i = 0; i < pixels.length; i++)
				{
					rgbPixels[i * 3] = pixels[i];
					rgbPixels[i * 3 + 1] = pixels[i];
					rgbPixels[i * 3 + 2] = pixels[i];
				}

				pixelDataLength = 3;
				pixels = rgbPixels;
			}

			if (properties.getGroupByChannel())
			{
				if (subtractMean)
				{
					double mean1 = 0, mean2 = 0, mean3 = 0;
					for (int j = 0; j < size; j++)
					{
						nextInput[j] = pixels[j * pixelDataLength + pixelDataLength - 1] & 0xFF;
						nextInput[j + size] = pixels[j * pixelDataLength + pixelDataLength - 2] & 0xFF;
						nextInput[j + size * 2] = pixels[j * pixelDataLength + pixelDataLength - 3] & 0xFF;
						mean1 += nextInput[j];
						mean2 += nextInput[j + size];
						mean3 += nextInput[j + size * 2];
					}

					float m1 = (float) (mean1 / size);
					float m2 = (float) (mean2 / size);
					float m3 = (float) (mean3 / size);
					for (int j = 0; j < size; j++)
					{
						nextInput[j] = (nextInput[j] - m1) / scaleColors;
						nextInput[j + size] = (nextInput[j + size] - m2) / scaleColors;
						nextInput[j + size * 2] = (nextInput[j + size * 2] - m3) / scaleColors;
					}
				} else if (meanArray != null)
				{
					for (int j = 0; j < size; j++)
					{
						nextInput[j] = (pixels[j * pixelDataLength + pixelDataLength - 1] & 0xFF) / scaleColors - meanArray[j];
						nextInput[j + size] = (pixels[j * pixelDataLength + pixelDataLength - 2] & 0xFF) / scaleColors - meanArray[j + size];
						nextInput[j + size * 2] = (pixels[j * pixelDataLength + pixelDataLength - 3] & 0xFF) / scaleColors - meanArray[j + size * 2];
					}
				} else
				{
					for (int j = 0; j < size; j++)
					{
						nextInput[j] = (pixels[j * pixelDataLength + pixelDataLength - 1] & 0xFF) / scaleColors;
						nextInput[j + size] = (pixels[j * pixelDataLength + pixelDataLength - 2] & 0xFF) / scaleColors;
						nextInput[j + size * 2] = (pixels[j * pixelDataLength + pixelDataLength - 3] & 0xFF) / scaleColors;
					}
				}
			} else
			{
				if (subtractMean)
				{
					double mean1 = 0, mean2 = 0, mean3 = 0;

					for (int j = 0; j < size; j++)
					{
						nextInput[j * 3] = pixels[j * pixelDataLength + pixelDataLength - 1] & 0xFF;
						nextInput[j * 3 + 1] = pixels[j * pixelDataLength + pixelDataLength - 2] & 0xFF;
						nextInput[j * 3 + 2] = pixels[j * pixelDataLength + pixelDataLength - 3] & 0xFF;
						mean1 += nextInput[j * 3];
						mean2 += nextInput[j * 3 + 1];
						mean3 += nextInput[j * 3 + 2];
					}

					float m1 = (float) (mean1 / size);
					float m2 = (float) (mean2 / size);
					float m3 = (float) (mean3 / size);
					for (int j = 0; j < size; j++)
					{
						nextInput[j * 3] = (nextInput[j * 3] - m1) / scaleColors;
						nextInput[j * 3 + 1] = (nextInput[j * 3 + 1] - m2) / scaleColors;
						nextInput[j * 3 + 2] = (nextInput[j * 3 + 2] - m3) / scaleColors;
					}
				} else if (meanArray != null)
				{
					for (int j = 0; j < size; j++)
					{
						nextInput[j * 3] = (pixels[j * pixelDataLength + pixelDataLength - 1] & 0xFF) / scaleColors - meanArray[j * 3];
						nextInput[j * 3 + 1] = (pixels[j * pixelDataLength + pixelDataLength - 2] & 0xFF) / scaleColors - meanArray[j * 3 + 1];
						nextInput[j * 3 + 2] = (pixels[j * pixelDataLength + pixelDataLength - 3] & 0xFF) / scaleColors - meanArray[j * 3 + 2];
					}
				} else
				{
					for (int j = 0; j < size; j++)
					{
						nextInput[j * 3] = (pixels[j * pixelDataLength + pixelDataLength - 1] & 0xFF) / scaleColors;
						nextInput[j * 3 + 1] = (pixels[j * pixelDataLength + pixelDataLength - 2] & 0xFF) / scaleColors;
						nextInput[j * 3 + 2] = (pixels[j * pixelDataLength + pixelDataLength - 3] & 0xFF) / scaleColors;
					}
				}
			}
		}

		if (properties.getSubtractValue() != null)
		{
			float v = properties.getSubtractValue();
			for (int i = 0; i < nextInput.length; i++)
			{
				nextInput[i] -= v;
			}
		}

		return nextInput;
	}

	/**
	 * the array is moved to free arrays
	 */
	protected void freeArray(float[] array)
	{
		synchronized (lock)
		{
			BufferedImage im = arrayToAugmented.get(array);
			if (im != null)
			{
				arrayToAugmented.remove(array);
				augmentedToRaw.remove(im);
				freeArrays.add(array);
			}
		}
	}

	@Override
	public void reset()
	{
		synchronized (lock)
		{
			super.reset();

			augmentedToRaw.clear();
			rawMBPosition.clear();
			arrayToAugmented.clear();
			freeArrays.clear();
			currentArrays.clear();

			if (thread != null)
			{
				thread.setRunning(false);
				thread = null;
			}

			if (values != null)
			{
				values.clear();
				values = null;
			}
		}
	}

	public BlockingQueue<float[]> getValues()
	{
		if (values == null)
		{
			values = new ArrayBlockingQueue<float[]>(properties.getAugmentedImagesBufferSize());
		}

		return values;
	}

	public ImageInputProviderProperties getProperties()
	{
		return properties;
	}

	public void setProperties(ImageInputProviderProperties properties)
	{
		this.properties = properties;
	}

	public Map<BufferedImage, BufferedImage> getAugmentedToRaw()
	{
		return augmentedToRaw;
	}

	public Map<float[], BufferedImage> getFloatToAugmented()
	{
		return arrayToAugmented;
	}

	public Object getLock()
	{
		return lock;
	}

	@Override
	public int getInputSize()
	{
		int result = 1;
		if (properties.getAugmentStrategy() != null)
		{
			result = properties.getAugmentStrategy().getSize();
		}

		return result;
	}

	/**
	 * This thread populates the images in the background
	 */
	private class BackgroundThread extends Thread
	{
		private boolean isRunning;

		public BackgroundThread()
		{
			super();
		}

		@Override
		public void run()
		{
			while (isRunning)
			{
				createAugmentedImages(getNextRawImages(), augmentedToRaw);
				createAugmentedArrays(augmentedToRaw, arrayToAugmented, values);
			}
		}

		public void setRunning(boolean isRunning)
		{
			this.isRunning = isRunning;
		}
	}

	/**
	 * Properties class for better visibility (too many member variables otherwise
	 */
	public static class ImageInputProviderProperties extends Properties
	{

		private static final long serialVersionUID = 1L;

		public ImageInputProviderProperties()
		{
			super();
			init();
		}

		public ImageInputProviderProperties(int initialCapacity, float loadFactor)
		{
			super(initialCapacity, loadFactor);
			init();
		}

		public ImageInputProviderProperties(int initialCapacity)
		{
			super(initialCapacity);
			init();
		}

		public ImageInputProviderProperties(Map<? extends String, ? extends Object> m)
		{
			super(m);
			init();
		}

		private void init()
		{
			setScaleColors(true);
			setGroupByChannel(true);
			setAugmentedImagesBufferSize(1);
			setParallelPreprocessing(false);
			setBackgroundThread(true);
			setSubtractMean(false);
		}

		/**
		 * is the image grayscale (adjusted automatically if not explicitly set)
		 */
		public Boolean getIsGrayscale()
		{
			return getParameter("isGrayscale");
		}

		public void setIsGrayscale(Boolean isGrayscale)
		{
			setParameter("isGrayscale", isGrayscale);
		}

		/**
		 * scale colors in the [0,1] range
		 */
		public boolean getScaleColors()
		{
			return (Integer) getParameter("scaleColors") == 255 ? true : false;
		}

		public void setScaleColors(boolean scaleColors)
		{
			setParameter("scaleColors", scaleColors ? 255 : 1);
		}

		/**
		 * group output by pixel (3 pixel colors sequential) or by channel (one
		 * pixel channel sequential)
		 */
		public boolean getGroupByChannel()
		{
			return getParameter("groupByChannel");
		}

		public void setGroupByChannel(boolean groupByChannel)
		{
			setParameter("groupByChannel", groupByChannel);
		}

		/**
		 * subtract the mean value for each image
		 */
		public boolean getSubtractMean()
		{
			return getParameter("subtractMean");
		}

		public void setSubtractMean(boolean subtractMean)
		{
			setParameter("subtractMean", subtractMean);
		}
		
		/**
		 * subtract the given array from the input array (for example mean values array)
		 */
		public float[] getSubtractArray()
		{
			return getParameter("subtractArray");
		}
		
		public void setSubtractArray(float[] subtractArray)
		{
			setParameter("subtractArray", subtractArray);
		}

		/**
		 * subtract the mean value for each image
		 */
		public Float getSubtractValue()
		{
			return getParameter("subtractValue");
		}

		public void setSubtractValue(Float subtractValue)
		{
			setParameter("subtractValue", subtractValue);
		}

		public int getImagesBulkSize()
		{
			return getParameter("imagesBulkSize");
		}

		public void setImagesBulkSize(int imagesBulkSize)
		{
			setParameter("imagesBulkSize", imagesBulkSize);
		}

		/**
		 * how many pre-processed images the buffer has
		 */
		public int getAugmentedImagesBufferSize()
		{
			return getParameter("augmentedImagesBufferSize");
		}

		public void setAugmentedImagesBufferSize(int augmentedImagesBufferSize)
		{
			setParameter("augmentedImagesBufferSize", augmentedImagesBufferSize);
		}

		/**
		 * use parallel preprocessing for performance optimization
		 */
		public boolean getParallelPreprocessing()
		{
			return getParameter("parallelPreprocessing");
		}

		public void setParallelPreprocessing(boolean parallelPreprocessing)
		{
			setParameter("parallelPreprocessing", parallelPreprocessing);
		}

		/**
		 * use background thread to process images in the background (while the gpu is working)
		 */
		public boolean getBackgroundThread()
		{
			return getParameter("backgroundThread");
		}

		public void setBackgroundThread(boolean backgroundThread)
		{
			setParameter("backgroundThread", backgroundThread);
		}

		/**
		 * resize strategy
		 */
		public ImageResizeStrategy getResizeStrategy()
		{
			return getParameter("resizeStrategy");
		}

		public void setResizeStrategy(ImageResizeStrategy resizeStrategy)
		{
			setParameter("resizeStrategy", resizeStrategy);
		}

		/**
		 * augment strategy
		 */
		public ImageAugmentStrategy getAugmentStrategy()
		{
			return getParameter("augmentStrategy");
		}

		public void setAugmentStrategy(ImageAugmentStrategy augmentStrategy)
		{
			setParameter("augmentStrategy", augmentStrategy);
		}
	}
}
