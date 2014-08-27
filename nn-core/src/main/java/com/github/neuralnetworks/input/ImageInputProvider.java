package com.github.neuralnetworks.input;

import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import com.github.neuralnetworks.training.TrainingInputProviderImpl;
import com.github.neuralnetworks.util.Properties;

/**
 * Base class for image input providers. Supported image augmentation operations: all affine transforms and cropping. Specify maxRotationAngle explicitly!
 */
public abstract class ImageInputProvider extends TrainingInputProviderImpl {

    private static final long serialVersionUID = 1L;

    protected float[] nextInput;

    protected ImageInputProviderProperties properties;

    /**
     * raw images
     */
    private List<BufferedImage> rawImages;

    /**
     * augmented images (resize, translation, cropping etc)
     */
    private List<BufferedImage> augmentedImages;

    public ImageInputProvider() {
	this(null);
    }

    public ImageInputProvider(InputConverter inputConverter) {
	super(inputConverter);
	rawImages = Collections.synchronizedList(new ArrayList<>());
	augmentedImages = Collections.synchronizedList(new ArrayList<>());
	properties = new ImageInputProviderProperties();
    }

    @Override
    public float[] getNextInput() {
	populateAugmentedImagesBuffer();

	BufferedImage image = augmentedImages.remove(0);

	byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
	int size = image.getWidth() * image.getHeight();
	int pixelDataLength = pixels.length / size;

	if (nextInput == null) {
	    // check for grayscale if not explicitly set
	    if (properties.getIsGrayscale() == null) {
		properties.setIsGrayscale(true);
		for (int i = 0; i < pixels.length; i += pixelDataLength) {
		    if (pixels[i + pixelDataLength - 1] != pixels[i + pixelDataLength - 2] || pixels[i + pixelDataLength - 2] != pixels[i + pixelDataLength - 3]) {
			properties.setIsGrayscale(false);
			break;
		    }
		}
	    }

	    nextInput = new float[size * (properties.getIsGrayscale() ? 1 : 3)];
	}

	float scaleColors = properties.getScaleColors() ? 255 : 1;
	if (properties.getIsGrayscale()) {
	    for (int i = 0; i < size; i++) {
		nextInput[i] = (pixels[i * pixelDataLength + pixelDataLength - 1] & 0xFF) / scaleColors;
	    }
	} else if (properties.getGroupByChannel()) {
	    for (int i = 0; i < size; i++) {
		nextInput[i] = (pixels[i * pixelDataLength + pixelDataLength - 1] & 0xFF) / scaleColors;
		nextInput[i + size] = (pixels[i * pixelDataLength + pixelDataLength - 2] & 0xFF) / scaleColors;
		nextInput[i + size * 2] = (pixels[i * pixelDataLength + pixelDataLength - 3] & 0xFF) / scaleColors;
	    }
	} else {
	    for (int i = 0; i < size; i++) {
		nextInput[i * 3] = (pixels[i * pixelDataLength + pixelDataLength - 3] & 0xFF) / scaleColors;
		nextInput[i * 3 + 1] = (pixels[i * pixelDataLength + pixelDataLength - 2] & 0xFF) / scaleColors;
		nextInput[i * 3 + 2] = (pixels[i * pixelDataLength + pixelDataLength - 1] & 0xFF) / scaleColors;
	    }
	}

	return nextInput;
    }

    /**
     * @return next image from the set
     */
    protected abstract BufferedImage getNextImage();

    /**
     * when the augmentedImages buffer is empty this method populates it again
     */
    protected void populateAugmentedImagesBuffer() {
	if (augmentedImages.size() == 0) {
	    if (requireAugmentation()) {
		IntStream.of(properties.getAugmentedImagesBufferSize() - rawImages.size()).forEach(i -> rawImages.add(getNextImage()));
		Stream<BufferedImage> stream = properties.getParallelPreprocessing() ? rawImages.parallelStream() : rawImages.stream();
		stream.forEach(i -> {
		    int index = rawImages.indexOf(i);
		    Random r = properties.getRandom();
		    if (properties.getCropX() != 0 || properties.getCropY() != 0) {
			i = i.getSubimage(r.nextInt(properties.getCropX() + 1), r.nextInt(properties.getCropY() + 1), i.getWidth() - properties.getCropX(), i.getHeight() - properties.getCropY());
		    }

		    if (properties.getAffineTransform() != null) {
			AffineTransform currentAf = properties.getAffineTransform();
			AffineTransform af = new AffineTransform();
			af.scale(properties.getAffineTransform().getScaleX(), currentAf.getScaleY());

			if (currentAf.getTranslateX() != 0 || currentAf.getTranslateY() != 0) {
			    af.translate(r.nextDouble() * currentAf.getTranslateX(), r.nextDouble() * currentAf.getTranslateY());
			}

			if (currentAf.getShearX() != 0 || currentAf.getShearY() != 0) {
			    af.shear(r.nextDouble() * currentAf.getShearX(), r.nextDouble() * currentAf.getShearY());
			}

			if (properties.getMaxRotationAngle() != 0) {
			    af.rotate(r.nextDouble() / 360);
			}

			AffineTransformOp op = new AffineTransformOp(af, AffineTransformOp.TYPE_BILINEAR);
			BufferedImage dest = new BufferedImage((int) (i.getWidth() / af.getScaleX()), (int) (i.getHeight() / af.getScaleY()), BufferedImage.TYPE_3BYTE_BGR);
			op.filter(i, dest);
			augmentedImages.add(index, dest);
		    } else {
			augmentedImages.add(index, i);
		    }
		});
	    } else {
		IntStream.of(properties.getAugmentedImagesBufferSize() - rawImages.size()).forEach(i -> augmentedImages.add(getNextImage()));
	    }

	    rawImages.clear();
	}
    }

    /**
     * @return whether a transformation is required based on the properties
     */
    protected boolean requireAugmentation() {
	return properties.getCropX() != 0 || properties.getCropY() != 0 || properties.getAffineTransform() != null;
    }

    public ImageInputProviderProperties getProperties() {
        return properties;
    }

    public void setProperties(ImageInputProviderProperties properties) {
        this.properties = properties;
    }

    /**
     * Properties class for better visibility (too many member variables otherwise
     */
    public static class ImageInputProviderProperties extends Properties {

	private static final long serialVersionUID = 1L;

	public ImageInputProviderProperties() {
	    super();
	    init();
	}

	public ImageInputProviderProperties(int initialCapacity, float loadFactor) {
	    super(initialCapacity, loadFactor);
	    init();
	}

	public ImageInputProviderProperties(int initialCapacity) {
	    super(initialCapacity);
	    init();
	}

	public ImageInputProviderProperties(Map<? extends String, ? extends Object> m) {
	    super(m);
	    init();
	}

	private void init() {
	    setScaleColors(true);
	    setGroupByChannel(true);
	    setAugmentedImagesBufferSize(1);
	    setParallelPreprocessing(false);
	    setCropX(0);
	    setCropY(0);
	    setMaxRotationAngle(0);
	    setUseRandomOrder(true);
	    setRandom(new Random());
	}

	public AffineTransform getAffineTransform() {
	    return getParameter("affineTransform");
	}

	public void setAffineTransform(AffineTransform affineTransform) {
	    setParameter("affineTransform", affineTransform);
	}

	/**
	 * is the image grayscale (adjusted automatically if not explicitly set)
	 */
	public Boolean getIsGrayscale() {
	    return getParameter("isGrayscale");
	}

	public void setIsGrayscale(Boolean isGrayscale) {
	    setParameter("isGrayscale", isGrayscale);
	}

	/**
	 * scale colors in the [0,1] range
	 */
	public boolean getScaleColors() {
	    return (Integer) getParameter("scaleColors") == 255 ? true : false;
	}

	public void setScaleColors(boolean scaleColors) {
	    setParameter("scaleColors", scaleColors ? 255 : 1);
	}

	/**
	 * group output by pixel (3 pixel colors sequential) or by channel (one
	 * pixel channel sequential)
	 */
	public boolean getGroupByChannel() {
	    return getParameter("groupByChannel");
	}

	public void setGroupByChannel(boolean groupByChannel) {
	    setParameter("groupByChannel", groupByChannel);
	}

	/**
	 * how many pre-processed images the buffer has
	 */
	public int getAugmentedImagesBufferSize() {
	    return getParameter("augmentedImagesBufferSize");
	}

	public void setAugmentedImagesBufferSize(int augmentedImagesBufferSize) {
	    setParameter("augmentedImagesBufferSize", augmentedImagesBufferSize);
	}

	/**
	 * use parallel preprocessing for performance optimization
	 */
	public boolean getParallelPreprocessing() {
	    return getParameter("parallelPreprocessing");
	}

	public void setParallelPreprocessing(boolean parallelPreprocessing) {
	    setParameter("parallelPreprocessing", parallelPreprocessing);
	}

	/**
	 * Image crop X
	 */
	public int getCropX() {
	    return getParameter("cropX");
	}

	public void setCropX(int cropX) {
	    setParameter("cropX", cropX);
	}

	/**
	 * Image crop Y
	 */
	public int getCropY() {
	    return getParameter("cropY");
	}

	public void setCropY(int cropY) {
	    setParameter("cropY", cropY);
	}

	/**
	 * maximum rotation angle
	 */
	public double getMaxRotationAngle() {
	    return getParameter("maxRotationAngle");
	}

	public void setMaxRotationAngle(double maxRotationAngle) {
	    setParameter("maxRotationAngle", maxRotationAngle);
	}

	public boolean getUseRandomOrder() {
	    return getParameter("useRandomOrder");
	}

	public void setUseRandomOrder(boolean useRandomOrder) {
	    setParameter("useRandomOrder", useRandomOrder);
	}

	public Random getRandom() {
	    return getParameter("random");
	}

	public void setRandom(Random random) {
	    setParameter("random", random);
	}
    }
}
