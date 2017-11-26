package com.github.neuralnetworks.input.image;

import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.util.List;

public class VerticalFlipAugmentStrategy implements ImageAugmentStrategy
{
	private static final long serialVersionUID = 1L;

	@Override
	public void addAugmentedImages(List<BufferedImage> images)
	{
		for (int i = 0, l = images.size(); i < l; i++)
		{
			BufferedImage source = images.get(i);
			AffineTransform tx = AffineTransform.getScaleInstance(1, -1);
			tx.translate(0, -source.getHeight(null));
			AffineTransformOp op = new AffineTransformOp(tx, AffineTransformOp.TYPE_NEAREST_NEIGHBOR);
			images.add(op.filter(source, null));
		}
	}

	@Override
	public int getSize()
	{
		return 2;
	}
}
