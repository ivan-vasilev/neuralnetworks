package com.github.neuralnetworks.input;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

/**
 * ImageInputProvider that retrieves images from a file directory
 */
public class FileImageInputProvider extends ImageInputProvider {

    private static final long serialVersionUID = 1L;

    private File directory;
    private String[] files;
    private List<Integer> elementsOrder;
    private int currentEl;
    private Random random;

    public FileImageInputProvider(File directory) {
	this(null, null);
    }

    public FileImageInputProvider(InputConverter inputConverter, File directory) {
	super(inputConverter);
	this.directory = directory;
	this.files = directory.list();
    }

    @Override
    public int getInputSize() {
	return files.length * getRepeatImage();
    }

    @Override
    public float[] getNextTarget() {
	return null;
    }

    @Override
    protected BufferedImage getNextImage() {
	BufferedImage result = null;
	try {
	    result = ImageIO.read(new File(directory, files[currentEl]));
	} catch (IOException e) {
	    e.printStackTrace();
	}

	return result;
    }

    @Override
    public void beforeSample() {
	if (elementsOrder.size() == 0) {
	    resetOrder();
	}

	currentEl = elementsOrder.remove(random.nextInt(elementsOrder.size()));
    }

    @Override
    public void reset() {
	super.reset();
	resetOrder();
    }

    public void resetOrder() {
	elementsOrder = new ArrayList<Integer>(files.length);
	for (int i = 0; i < files.length; i++) {
	    elementsOrder.add(i);
	}
    }
}
