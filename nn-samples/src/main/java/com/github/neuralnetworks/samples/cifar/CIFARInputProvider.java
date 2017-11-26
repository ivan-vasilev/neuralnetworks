package com.github.neuralnetworks.samples.cifar;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.input.image.ImageInputProvider;

/**
 * Input provider for the CIFAR-10 and CIFAR-100 datasets. Requires location of the CIFAR images
 * files (not included in the library). Do not use this class directly, but use the subclasses instead
 * Experimental
 */
public abstract class CIFARInputProvider extends ImageInputProvider
{
	private static final long serialVersionUID = 1L;

	protected static final int RAW_INPUT_SAMPLE_SIZE = 32 * 32 * 3;

	protected transient RandomAccessFile files[];
	protected int labelSize;
	protected int rawInputSize;
	protected byte[] nextInputRaw;
	protected BufferedImage[] images;
	protected int currentRawImage;
	protected Map<Integer, Integer> imageToTarget;
	protected int inputDimensions;

	private CIFARInputProvider(int inputDimensions)
	{
		super();
		this.inputDimensions = inputDimensions;
		this.nextInputRaw = new byte[RAW_INPUT_SAMPLE_SIZE];
		this.properties.setImagesBulkSize(1);
		this.imageToTarget = new HashMap<>();
	}

	@Override
	public int getInputSize()
	{
		return rawInputSize * super.getInputSize();
	}

	public abstract int getTargetSize();

	@Override
	public List<BufferedImage> getNextRawImages()
	{
		if (images == null) 
		{
			images = new BufferedImage[properties.getImagesBulkSize()];
		}

		for (int i = 0; i < properties.getImagesBulkSize(); i++, currentRawImage++)
		{
			// read image
			RandomAccessFile f = files[(currentRawImage % rawInputSize) / (rawInputSize / files.length)];
			int id = currentRawImage % (rawInputSize / files.length);
			int target = 0;

			try
			{
				f.seek(id * (RAW_INPUT_SAMPLE_SIZE + labelSize));
				if (labelSize > 1)
				{
					f.readUnsignedByte();
				}

				target = f.readUnsignedByte();

				f.readFully(nextInputRaw);
			} catch (IOException e)
			{
				e.printStackTrace();
			}

			// populate
			// initialize BufferedImage as 3BYTE_BGR to safe heap (?)
			images[i] = new BufferedImage(32, 32, BufferedImage.TYPE_3BYTE_BGR);
			byte[] pixels = ((DataBufferByte) images[i].getRaster().getDataBuffer()).getData();

			// Cifar image input is stored as 32x32 (1024) pixels as RRR...GGG...BBB...
			// - has to be flipped while storing in member 'pixels'
			for (int j = 0; j < 1024; j++)
			{
				pixels[j * 3] = nextInputRaw[1024 * 2 + j];
				pixels[j * 3 + 1] = nextInputRaw[1024 + j];
				pixels[j * 3 + 2] = nextInputRaw[j];
			}

			// target
			imageToTarget.put(images[i].hashCode(), target);
		}

		return Arrays.asList(images);
	}

	@Override
	protected float[] getNextTarget(BufferedImage image) 
	{
		float[] target = new float[getTargetSize()];
		target[imageToTarget.get(image.hashCode())] = 1;

		return target;
	}

	@Override
	public void reset()
	{
		super.reset();
		currentRawImage = 0;
	}

	@Override
	public int getInputDimensions()
	{
		return inputDimensions;
	}

	public void setInputDimensions(int inputDimensions)
	{
		this.inputDimensions = inputDimensions;
	}

	@Override
	public int getTargetDimensions()
	{
		return 10;
	}

	public static class CIFAR10TrainingInputProvider extends CIFARInputProvider
	{

		private static final long serialVersionUID = 1L;

		private String directory;

		public CIFAR10TrainingInputProvider(String directory)
		{
			this(directory, RAW_INPUT_SAMPLE_SIZE);
		}

		/**
		 * @param directory - the folder where the CIFAR files are located
		 * @param inputDimensions - width * height * 3
		 */
		public CIFAR10TrainingInputProvider(String directory, int inputDimensions)
		{
			super(inputDimensions);

			this.directory = directory;
			this.labelSize = 1;
			this.rawInputSize = 50000;

			initFiles();
		}

		private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException
		{
			stream.defaultReadObject();
			initFiles();
		}

		private void initFiles()
		{
			files = new RandomAccessFile[5];

			try
			{
				if (!directory.endsWith(File.separator))
				{
					directory += File.separator;
				}

				files[0] = new RandomAccessFile(directory + "data_batch_1.bin", "r");
				files[1] = new RandomAccessFile(directory + "data_batch_2.bin", "r");
				files[2] = new RandomAccessFile(directory + "data_batch_3.bin", "r");
				files[3] = new RandomAccessFile(directory + "data_batch_4.bin", "r");
				files[4] = new RandomAccessFile(directory + "data_batch_5.bin", "r");
			} catch (FileNotFoundException e)
			{
				e.printStackTrace();
			}
		}

		public void setup(String directory, int labelSize, int rawInputSize, int inputDimensions, String... aFiles)
		{
			files = new RandomAccessFile[aFiles.length];

			this.inputDimensions = inputDimensions;
			this.nextInputRaw = new byte[RAW_INPUT_SAMPLE_SIZE];
			this.properties.setImagesBulkSize(1);
			this.imageToTarget = new HashMap<>();

			this.directory = directory;
			this.labelSize = labelSize;
			this.rawInputSize = rawInputSize;

			try
			{
				if (!directory.endsWith(File.separator))
				{
					directory += File.separator;
				}

				for (int i = 0; i < aFiles.length; i++) {
					files[i] = new RandomAccessFile(directory + aFiles[i], "r");
				}
			} catch (FileNotFoundException e)
			{
				e.printStackTrace();
			}
		}

		@Override
		public int getTargetSize()
		{
			return 10;
		}
	}

	public static class CIFAR10TestingInputProvider extends CIFARInputProvider
	{

		private static final long serialVersionUID = 1L;

		private String directory;

		public CIFAR10TestingInputProvider(String directory)
		{
			this(directory, RAW_INPUT_SAMPLE_SIZE);
		}

		/**
		 * @param directory - the folder where the CIFAR files are located
		 * @param inputDimensions - width * height * 3
		 */
		public CIFAR10TestingInputProvider(String directory, int inputDimensions)
		{
			super(inputDimensions);

			this.directory = directory;
			this.labelSize = 1;
			this.rawInputSize = 10000;

			initFiles();
		}

		private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException
		{
			stream.defaultReadObject();
			initFiles();
		}

		private void initFiles()
		{
			this.files = new RandomAccessFile[1];

			try
			{
				if (!directory.endsWith(File.separator))
				{
					directory += File.separator;
				}

				files[0] = new RandomAccessFile(directory + "test_batch.bin", "r");
			} catch (FileNotFoundException e)
			{
				e.printStackTrace();
			}
		}

		@Override
		public int getTargetSize()
		{
			return 10;
		}
	}

	public static class CIFAR100TrainingInputProvider extends CIFARInputProvider
	{

		private static final long serialVersionUID = 1L;

		private String directory;

		public CIFAR100TrainingInputProvider(String directory)
		{
			this(directory, RAW_INPUT_SAMPLE_SIZE);
		}

		/**
		 * @param directory - the folder where the CIFAR files are located
		 * @param inputDimensions - width * height * 3
		 */
		public CIFAR100TrainingInputProvider(String directory, int inputDimensions)
		{
			super(inputDimensions);

			this.directory = directory;
			this.labelSize = 2;
			this.rawInputSize = 50000;

			initFiles();
		}

		private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException
		{
			stream.defaultReadObject();
			initFiles();
		}

		private void initFiles()
		{
			files = new RandomAccessFile[5];

			try
			{
				if (!directory.endsWith(File.separator))
				{
					directory += File.separator;
				}

				files[0] = new RandomAccessFile(directory + "data_batch_1.bin", "r");
				files[1] = new RandomAccessFile(directory + "data_batch_2.bin", "r");
				files[2] = new RandomAccessFile(directory + "data_batch_3.bin", "r");
				files[3] = new RandomAccessFile(directory + "data_batch_4.bin", "r");
				files[4] = new RandomAccessFile(directory + "data_batch_5.bin", "r");
			} catch (FileNotFoundException e)
			{
				e.printStackTrace();
			}
		}

		@Override
		public int getTargetSize()
		{
			return 100;
		}
	}

	public static class CIFAR100TestingInputProvider extends CIFARInputProvider
	{

		private static final long serialVersionUID = 1L;

		private String directory;

		public CIFAR100TestingInputProvider(String directory)
		{
			this(directory, RAW_INPUT_SAMPLE_SIZE);
		}

		/**
		 * @param directory - the folder where the CIFAR files are located
		 * @param inputDimensions - width * height * 3
		 */
		public CIFAR100TestingInputProvider(String directory, int inputDimensions)
		{
			super(inputDimensions);

			this.directory = directory;
			this.labelSize = 2;
			this.rawInputSize = 10000;

			initFiles();
		}

		private void initFiles()
		{
			this.files = new RandomAccessFile[1];

			try
			{
				if (!directory.endsWith(File.separator))
				{
					directory += File.separator;
				}

				files[0] = new RandomAccessFile(directory + "test_batch.bin", "r");
			} catch (FileNotFoundException e)
			{
				e.printStackTrace();
			}
		}

		@Override
		public int getTargetSize()
		{
			return 100;
		}
	}
}
