package com.github.neuralnetworks.input.image;

import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.util.List;

public interface ImageAugmentStrategy extends Serializable
{
	public void addAugmentedImages(List<BufferedImage> images);
	public int getSize();
}
