package com.github.neuralnetworks.calculation.operations;

import org.junit.Test;

import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.test.AbstractTest;

/**
 * Created by chass on 28.11.14.
 */
public class SoftmaxFunctionTest extends AbstractTest
{

	@Test
	public void testSoftMaxPrecision()
	{
		testSoftMaxPrecision(Runtime.CPU_SEQ);
		testSoftMaxPrecision(Runtime.OPENCL);
	}

	public void testSoftMaxPrecision(Runtime runtime)
	{
		configureGlobalRuntimeEnvironment(runtime);

		int[] globalDimensions = new int[2];
		globalDimensions[0] = 100;
		globalDimensions[1] = 10;

		int[][] globalDimensionsLimit = new int[2][2];
		globalDimensionsLimit[0][0] = 0;
		globalDimensionsLimit[1][0] = 9;
		globalDimensionsLimit[1][1] = 9;
		globalDimensionsLimit[0][1] = 0;

		float[] elements = new float[1000];
		elements[0] = 0.9999999f;
		elements[1] = 0.00000002f;
		for (int i = 1; i < 10; i++)
		{
			elements[i] = 0.00000001f;
		}

		Matrix tensor = new Matrix(0, elements, globalDimensions, globalDimensionsLimit);
		TensorFunction softmax = OperationsFactory.softmaxFunction();
		softmax.value(tensor);
	}

}
