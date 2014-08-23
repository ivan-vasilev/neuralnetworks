package com.github.neuralnetworks.input;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import com.github.neuralnetworks.training.TrainingInputProviderImpl;

/**
 * Base class for image input providers.
 */
public abstract class ImageInputProvider extends TrainingInputProviderImpl {

    private static final long serialVersionUID = 1L;

    private float[] nextInput;

    /**
     * is the image grayscale (adjusted automatically if not explicitly set)
     */
    private Boolean isGrayscale;

    /**
     * scale colors in the [0,1] range
     */
    private boolean scaleColors;

    /**
     * repeat the same image several times (when image transformations are
     * applied)
     */
    private int repeatImage;
    private int currentRepetition;

    /**
     * group output by pixel (3 pixel colors sequential) or by channel (one
     * pixel channel sequential)
     */
    private boolean groupByChannel;

    public ImageInputProvider() {
	this(null);
    }

    public ImageInputProvider(InputConverter inputConverter) {
	super(inputConverter);
	scaleColors = true;
	repeatImage = 1;
	currentRepetition = 0;
    }

    @Override
    public float[] getNextInput() {
	BufferedImage image = getNextImage();
	byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
	int size = image.getWidth() * image.getHeight();
	int pixelDataLength = pixels.length / size;

	if (nextInput == null) {
	    // check for grayscale if not explicitly set
	    if (isGrayscale == null) {
		isGrayscale = true;
		for (int i = 0; i < pixels.length; i += pixelDataLength) {
		    if (pixels[i + pixelDataLength - 1] != pixels[i + pixelDataLength - 2] || pixels[i + pixelDataLength - 2] != pixels[i + pixelDataLength - 3] || pixels[i + pixelDataLength - 3] != pixels[i + pixelDataLength - 4]) {
			isGrayscale = false;
			break;
		    }
		}
	    }

	    nextInput = new float[size * (isGrayscale ? 1 : 3)];
	}

	// if the current repetitions are over proceed with the next image
	if (repeatImage == ++currentRepetition) {
	    currentRepetition = 0;
	    int scaleFactor = scaleColors ? 255 : 1;
	    if (isGrayscale) {
		for (int i = 0; i < size; i++) {
		    nextInput[i] = (pixels[i * pixelDataLength + 1] & 0xFF) / scaleFactor;
		}
	    } else {
		// check pixel groups
		int pixelDistance = groupByChannel ? size : 1;
		for (int i = 0; i < size; i++) {
		    nextInput[i] = (pixels[i * pixelDataLength] & 0xFF) / scaleFactor;
		    nextInput[i + pixelDistance] = (pixels[i * pixelDataLength + 1] & 0xFF) / scaleFactor;
		    nextInput[i + pixelDistance] = (pixels[i * pixelDataLength + 2] & 0xFF) / scaleFactor;
		}
	    }
	}

	return nextInput;
    }

    /**
     * @return next image from the set
     */
    protected abstract BufferedImage getNextImage();

    public Boolean getIsGrayscale() {
	return isGrayscale;
    }

    public void setIsGrayscale(Boolean isGrayscale) {
	this.isGrayscale = isGrayscale;
    }

    public boolean getScaleColors() {
	return scaleColors;
    }

    public void setScaleColors(boolean scaleColors) {
	this.scaleColors = scaleColors;
    }

    public int getRepeatImage() {
	return repeatImage;
    }

    public void setRepeatImage(int repeatImage) {
	this.repeatImage = repeatImage;
    }

    public boolean isGroupByChannel() {
	return groupByChannel;
    }

    public void setGroupByChannel(boolean groupByChannel) {
	this.groupByChannel = groupByChannel;
    }
}
