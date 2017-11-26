package com.github.neuralnetworks.input.image;

import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.util.List;

public class HorizontalFlipAugmentStrategy implements ImageAugmentStrategy
{
	private static final long serialVersionUID = 1L;

	@Override
	public void addAugmentedImages(List<BufferedImage> images)
	{
		for (int i = 0, l = images.size(); i < l; i++)
		{
			BufferedImage source = images.get(i);
			AffineTransform tx = AffineTransform.getScaleInstance(-1.0, 1.0);
			tx.translate(-source.getWidth(), 0);
			AffineTransformOp tr = new AffineTransformOp(tx, AffineTransformOp.TYPE_NEAREST_NEIGHBOR);
			images.add(tr.filter(source, null));
		}
	}

	@Override
	public int getSize()
	{
		return 2;
	}
}
