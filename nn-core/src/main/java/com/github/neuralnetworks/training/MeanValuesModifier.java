package com.github.neuralnetworks.training;

import java.io.File;
import java.util.stream.IntStream;

import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.util.Util;

/**
 * Subtracts the mean value based on an array in the file
 */
public class MeanValuesModifier implements TensorFunction
{
	private static final long serialVersionUID = 1L;

	private float[] meanValues;

	public MeanValuesModifier(File meanFile)
	{
		super();
		this.meanValues = Util.readFileIntoFloatArray(meanFile);
	}

	@Override
	public void value(Tensor inputOutput)
	{
		int sampleLength = inputOutput.getSize() / inputOutput.getDimensions()[0];
		IntStream.range(0, inputOutput.getSize()).parallel().forEach(i -> inputOutput.getElements()[inputOutput.getStartIndex() + i] -= meanValues[i % sampleLength]);
	}
}
