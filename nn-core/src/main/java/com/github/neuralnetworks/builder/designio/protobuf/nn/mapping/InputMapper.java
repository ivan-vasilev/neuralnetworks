package com.github.neuralnetworks.builder.designio.protobuf.nn.mapping;

import java.io.File;

import com.github.neuralnetworks.builder.designio.protobuf.nn.InputProtoBufWrapper;
import com.github.neuralnetworks.input.image.CompositeAugmentStrategy;
import com.github.neuralnetworks.input.image.DirectoryImageInputProvider;
import com.github.neuralnetworks.input.image.HorizontalFlipAugmentStrategy;
import com.github.neuralnetworks.input.image.ImageInputProvider;
import com.github.neuralnetworks.input.image.ImageResizeStrategy;
import com.github.neuralnetworks.input.image.RandomCropAugmentStrategy;
import com.github.neuralnetworks.input.image.RotateAllAugmentStrategy;
import com.github.neuralnetworks.input.image.VerticalFlipAugmentStrategy;
import com.github.neuralnetworks.training.TrainingInputProvider;

/**
 * @author tmey
 */
public class InputMapper
{


	public static TrainingInputProvider parseInput(InputProtoBufWrapper.InputData inputData)
	{

		if (!inputData.hasType())
		{
			throw new IllegalArgumentException("Type of input date required!");
		}

		TrainingInputProvider trainingInputProvider = null;

		switch (inputData.getType())
		{
		case DIRECTORY:
		{
			if (!inputData.hasPath())
			{
				throw new IllegalArgumentException("The parameter path must be set and valid!");
			}
			trainingInputProvider = new DirectoryImageInputProvider(new File(inputData.getPath()));
			break;
		}

		default:
			throw new IllegalArgumentException("Unknown input data type: " + inputData.getType());

			// TODO write types of input data
		}

		if (inputData.hasTransformation() && trainingInputProvider instanceof ImageInputProvider)
		{
			ImageInputProvider imageInputProvider = (ImageInputProvider) trainingInputProvider;
			InputProtoBufWrapper.TransformationParameter transformation = inputData.getTransformation();

			imageInputProvider.getProperties().setScaleColors(transformation.getScaleColor());
			imageInputProvider.getProperties().setSubtractMean(transformation.getSubstractMean());

			imageInputProvider.getProperties().setImagesBulkSize(transformation.getImgBulkSize());
			imageInputProvider.getProperties().setAugmentedImagesBufferSize(transformation.getAugmImgBufSize());
			imageInputProvider.getProperties().setBackgroundThread(transformation.getBackgroundThread());
			imageInputProvider.getProperties().setParallelPreprocessing(transformation.getParallelPreprocessing());


			if (transformation.hasAugmentationStrategy())
			{
				parseAugmentation(imageInputProvider, inputData.getTransformation().getAugmentationStrategy());
			}
			if (transformation.hasResizeStrategy())
			{
				parseResizeStrategy(imageInputProvider, inputData.getTransformation().getResizeStrategy());
			}

		}

		return trainingInputProvider;
	}


	private static void parseAugmentation(ImageInputProvider inputProvider, InputProtoBufWrapper.AugmentationStrategy augmentationStrategy)
	{
		if (inputProvider == null)
		{
			throw new IllegalArgumentException("inputProvider must be not null!");
		}

		if (augmentationStrategy == null)
		{
			return;
		}

		CompositeAugmentStrategy compositeAugmentStrategy = new CompositeAugmentStrategy();

		// produce x times more images
		if (augmentationStrategy.getRandomCrop())
		{
			if (augmentationStrategy.hasSubsamplingHeight())
			{
				throw new IllegalArgumentException("The subsampling height is required for random crop!");
			}
			if (augmentationStrategy.hasSubsamplingWidth())
			{
				throw new IllegalArgumentException("The subsampling width is required for random crop!");
			}
			if (augmentationStrategy.hasSubsamplingNumber())
			{
				throw new IllegalArgumentException("The subsampling number is required for random crop!");
			}

			compositeAugmentStrategy.addStrategy(new RandomCropAugmentStrategy(
					augmentationStrategy.getSubsamplingWidth(),
					augmentationStrategy.getSubsamplingHeight(),
					augmentationStrategy.getSubsamplingNumber()));
		}

		// produce 4 times more images
		if (augmentationStrategy.getRotateComplete())
		{
			compositeAugmentStrategy.addStrategy(new RotateAllAugmentStrategy());
		}

		// produce 2 times more images
		if (augmentationStrategy.getHorizontalFlip())
		{
			compositeAugmentStrategy.addStrategy(new HorizontalFlipAugmentStrategy());
		}
		if (augmentationStrategy.getVerticalFlip())
		{
			compositeAugmentStrategy.addStrategy(new VerticalFlipAugmentStrategy());
		}

		inputProvider.getProperties().setAugmentStrategy(compositeAugmentStrategy);

	}

	private static void parseResizeStrategy(ImageInputProvider inputProvider, InputProtoBufWrapper.ImageResizeStrategy resizeStrategy)
	{
		if (inputProvider == null)
		{
			throw new IllegalArgumentException("inputProvider must be not null!");
		}

		if (resizeStrategy == null)
		{
			return;
		}

		if (!resizeStrategy.hasSmallestDimension())
		{
			throw new IllegalArgumentException("Smallest dimension is missing for the resize strategy!");
		}

		ImageResizeStrategy imageResizeStrategy = new ImageResizeStrategy(
				convertResizeType(resizeStrategy.getType()),
				resizeStrategy.getSmallestDimension(),
				resizeStrategy.getStepResize());


		inputProvider.getProperties().setResizeStrategy(imageResizeStrategy);
	}

	private static ImageResizeStrategy.ResizeType convertResizeType(InputProtoBufWrapper.ImageResizeStrategy.ResizeType resizeType)
	{

		switch (resizeType)
		{
		case SMALLEST_DIMENSION_RECT:
			return ImageResizeStrategy.ResizeType.SMALLEST_DIMENSION_RECT;
		case SMALLEST_DIMENSION_SQUARE_CROP_MIDDLE:
			return ImageResizeStrategy.ResizeType.SMALLEST_DIMENSION_SQUARE_CROP_MIDDLE;
		case SMALLEST_DIMENSION_SQUARE_SIZE:
			return ImageResizeStrategy.ResizeType.SMALLEST_DIMENSION_SQUARE_RESIZE;
		default:
			throw new IllegalArgumentException("Unknown resize type: " + resizeType);
		}
	}

}
