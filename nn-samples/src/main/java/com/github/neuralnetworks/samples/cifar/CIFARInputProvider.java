package com.github.neuralnetworks.samples.cifar;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import com.github.neuralnetworks.input.ImageInputProvider;
import com.github.neuralnetworks.util.Util;

/**
 * Input provider for the CIFAR-10 and CIFAR-100 datasets. Requires location of the CIFAR images
 * files (not included in the library). Do not use this class directly, but use the subclasses instead
 */
public abstract class CIFARInputProvider extends ImageInputProvider {

    private static final long serialVersionUID = 1L;

    protected RandomAccessFile files[];
    protected int labelSize;
    protected int inputSize;
    protected byte[] nextInputRaw;
    protected float[] nextTarget;
    private List<Integer> elementsOrder;

    private CIFARInputProvider() {
	super();
	this.elementsOrder = new ArrayList<>();
	this.nextInputRaw = new byte[3072];
    }

    @Override
    public int getInputSize() {
	return inputSize;
    }

    @Override
    public float[] getNextTarget() {
	return nextTarget;
    }

    @Override
    public float[] getNextInput() {
	// if no transformations are required and the data is grouped by color
	// channel the code can be optimized
	if (!requireAugmentation() && getProperties().getGroupByChannel()) {
	    if (nextInput == null) {
		nextInput = new float[3072];
	    }

	    int scaleColors = getProperties().getScaleColors() ? 255 : 1;
	    for (int i = 0; i < nextInput.length; i++) {
		nextInput[i] = (nextInputRaw[i] & 0xFF) / scaleColors;
	    }

	    return nextInput;
	}

	return super.getNextInput();
    }

    @Override
    protected BufferedImage getNextImage() {
	BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_3BYTE_BGR);
	byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();

	for (int i = 0; i < 1024; i++) {
	    pixels[i * 3] = nextInputRaw[1024 * 2 + i];
	    pixels[i * 3 + 1] = nextInputRaw[1024 + i];
	    pixels[i * 3 + 2] = nextInputRaw[i];
	}

	return image;
    }

    @Override
    public void beforeSample() {
	if (elementsOrder.size() == 0) {
	    resetOrder();
	}

	int currentEl = elementsOrder.remove(getProperties().getUseRandomOrder() ? getProperties().getRandom().nextInt(elementsOrder.size()) : 0);
	int id = currentEl % (getInputSize() / files.length);

	RandomAccessFile f = files[currentEl / (getInputSize() / files.length)];

	try {
	    f.seek(id * (3072 + labelSize));
	    if (labelSize > 1) {
		f.readUnsignedByte();
	    }

	    Util.fillArray(nextTarget, 0);
	    nextTarget[f.readUnsignedByte()] = 1;

	    f.readFully(nextInputRaw);
	} catch (IOException e) {
	    e.printStackTrace();
	}
    }

    @Override
    public void reset() {
	super.reset();
	resetOrder();
    }

    public void resetOrder() {
	elementsOrder = new ArrayList<Integer>(getInputSize());
	for (int i = 0; i < getInputSize(); i++) {
	    elementsOrder.add(i);
	}
    }

    public static class CIFAR10TrainingInputProvider extends CIFARInputProvider {

	private static final long serialVersionUID = 1L;

	public CIFAR10TrainingInputProvider(String directory) {
	    super();

	    this.labelSize = 1;
	    this.inputSize = 50000;
	    this.nextTarget = new float[10];
	    this.files = new RandomAccessFile[5];

	    try {
		if (!directory.endsWith(File.separator)) {
		    directory += File.separator;
		}

		files[0] = new RandomAccessFile(directory + "data_batch_1.bin", "r");
		files[1] = new RandomAccessFile(directory + "data_batch_2.bin", "r");
		files[2] = new RandomAccessFile(directory + "data_batch_3.bin", "r");
		files[3] = new RandomAccessFile(directory + "data_batch_4.bin", "r");
		files[4] = new RandomAccessFile(directory + "data_batch_5.bin", "r");
	    } catch (FileNotFoundException e) {
		e.printStackTrace();
	    }
	}
    }

    public static class CIFAR10TestingInputProvider extends CIFARInputProvider {

	private static final long serialVersionUID = 1L;

	public CIFAR10TestingInputProvider(String directory) {
	    super();

	    this.labelSize = 1;
	    this.inputSize = 10000;
	    this.nextTarget = new float[10];
	    this.files = new RandomAccessFile[1];

	    try {
		if (!directory.endsWith(File.separator)) {
		    directory += File.separator;
		}

		files[0] = new RandomAccessFile(directory + "test_batch.bin", "r");
	    } catch (FileNotFoundException e) {
		e.printStackTrace();
	    }
	}
    }

    public static class CIFAR100TrainingInputProvider extends CIFARInputProvider {

	private static final long serialVersionUID = 1L;

	public CIFAR100TrainingInputProvider(String directory) {
	    super();

	    this.labelSize = 2;
	    this.inputSize = 50000;
	    this.nextTarget = new float[100];
	    this.files = new RandomAccessFile[5];

	    try {
		if (!directory.endsWith(File.separator)) {
		    directory += File.separator;
		}

		files[0] = new RandomAccessFile(directory + "data_batch_1.bin", "r");
		files[1] = new RandomAccessFile(directory + "data_batch_2.bin", "r");
		files[2] = new RandomAccessFile(directory + "data_batch_3.bin", "r");
		files[3] = new RandomAccessFile(directory + "data_batch_4.bin", "r");
		files[4] = new RandomAccessFile(directory + "data_batch_5.bin", "r");
	    } catch (FileNotFoundException e) {
		e.printStackTrace();
	    }
	}
    }

    public static class CIFAR100TestingInputProvider extends CIFARInputProvider {
	
	private static final long serialVersionUID = 1L;
	
	public CIFAR100TestingInputProvider(String directory) {
	    super();

	    this.labelSize = 2;
	    this.inputSize = 10000;
	    this.nextTarget = new float[100];
	    this.files = new RandomAccessFile[1];

	    try {
		if (!directory.endsWith(File.separator)) {
		    directory += File.separator;
		}

		files[0] = new RandomAccessFile(directory + "test_batch.bin", "r");
	    } catch (FileNotFoundException e) {
		e.printStackTrace();
	    }
	}
    }
}
