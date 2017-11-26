package com.github.neuralnetworks.input.image;

import static org.junit.Assert.assertEquals;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import com.github.neuralnetworks.input.image.SlidingWindowAugmentStrategy;
import com.github.neuralnetworks.test.AbstractTest;

public class SlidingWindowImageAugmentStrategyTest extends AbstractTest
{

	@Test
	public void test1()
	{
		BufferedImage im = new BufferedImage(6, 3, BufferedImage.TYPE_3BYTE_BGR);
		Random rand = new Random(123);
		for (int j = 0; j < im.getWidth(); j++)
		{
			for (int k = 0; k < im.getHeight(); k++)
			{
				int b = rand.nextInt(256);
				int g = rand.nextInt(256);
				int r = rand.nextInt(256);
				im.setRGB(j, k, (b << 16) | (g << 8) | r);
			}
		}

		SlidingWindowAugmentStrategy sl = new SlidingWindowAugmentStrategy(3, 2, 2, 1);
		List<BufferedImage> images = new ArrayList<>();
		images.add(im);
		sl.addAugmentedImages(images);

		assertEquals(4, images.size());
		images.forEach(i -> {
			assertEquals(3, i.getWidth());
			assertEquals(2, i.getHeight());

			int x = images.indexOf(i) / 2;
			int y = images.indexOf(i) % 2;

			assertEquals(im.getRGB(x * sl.getStrideX(), y * sl.getStrideY()), i.getRGB(0, 0));
			assertEquals(im.getRGB(x * sl.getStrideX() + 1, y * sl.getStrideY() + 1), i.getRGB(1, 1));
			assertEquals(im.getRGB(x * sl.getStrideX() + 2, y * sl.getStrideY()), i.getRGB(2, 0));
			assertEquals(im.getRGB(x * sl.getStrideX(), y * sl.getStrideY() + 1), i.getRGB(0, 1));
			assertEquals(im.getRGB(x * sl.getStrideX() + 1, y * sl.getStrideY()), i.getRGB(1, 0));
			assertEquals(im.getRGB(x * sl.getStrideX() + 2, y * sl.getStrideY() + 1), i.getRGB(2, 1));
		});
	}
}
