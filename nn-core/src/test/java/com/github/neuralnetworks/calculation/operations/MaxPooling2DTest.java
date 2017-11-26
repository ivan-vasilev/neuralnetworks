package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiMaxPooling2D;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.BackpropagationMaxPooling2D2;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLMaxPooling2DBP;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLMaxPooling2DConnectionCalculator;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;

/**
 * Created by chass on 25.11.14.
 */
public class MaxPooling2DTest extends AbstractTest
{


	@Test
	public void testRandomMaxPoolingFF()
	{

		long seed = 123456789;

		int inputSquareSize = 200;
		int inputFeatures = 3;
		int pollingAreaSquareSize = 5;
		int batchSize = 2;
		int stride = 3;
		int paddingSize = 4;

		Tensor seqResult = testRandomMaxPoolingFF(Runtime.CPU_SEQ, seed, inputSquareSize, inputFeatures, pollingAreaSquareSize, batchSize, stride, paddingSize);
		Tensor openclResult = testRandomMaxPoolingFF(Runtime.OPENCL, seed, inputSquareSize, inputFeatures, pollingAreaSquareSize, batchSize, stride, paddingSize);

		assertTrue(isEqual(seqResult, openclResult));

	}

	private Tensor testRandomMaxPoolingFF(Runtime runtime, long seed, int inputSquareSize, int inputFeatures, int pollingAreaSquareSize, int batchSize, int stride, int outputPaddingSize)
	{

		configureGlobalRuntimeEnvironment(runtime);
		Random random = new Random(seed);

		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), inputSquareSize, inputSquareSize, pollingAreaSquareSize, pollingAreaSquareSize, inputFeatures, stride, stride,
				outputPaddingSize, outputPaddingSize);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ConnectionCalculator calc = OperationsFactory.maxPooling2D();

		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(calc instanceof OpenCLMaxPooling2DConnectionCalculator);
		} else
		{
			assertTrue(calc instanceof AparapiMaxPooling2D);
		}


		ValuesProvider vp = TensorFactory.tensorProvider(c, batchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

		float[] src = new float[inputSquareSize * inputSquareSize * inputFeatures * batchSize];
		for (int i = 0; i < src.length; i++)
		{
			src[i] = random.nextFloat() * 255;
		}

		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);
		calc.calculate(connections, vp, c.getOutputLayer());

		int outputSquareSizeNoPadding = c.getOutputFeatureMapRows();
		Tensor o = vp.get(c.getOutputLayer(), batchSize, inputFeatures, outputSquareSizeNoPadding, outputSquareSizeNoPadding); // searches for a tensor for this layer which has this dimentions
		assertNotNull(o);

		// test padding
		Tensor oPadding = vp.get(c.getOutputLayer(), batchSize, inputFeatures, outputSquareSizeNoPadding + (2 * outputPaddingSize), outputSquareSizeNoPadding + (2 * outputPaddingSize)); // searches for a
																																																																																											// tensor for this
																																																																																											// layer which has
																																																																																											// this dimentions
		assertNotNull(oPadding);

		for (int batch = 0; batch < batchSize; batch++)
		{
			for (int feat = 0; feat < inputFeatures; feat++)
			{
				for (int row = 0; row < outputSquareSizeNoPadding + (2 * outputPaddingSize); row++)
				{
					if (row < outputPaddingSize || row > (outputSquareSizeNoPadding + outputPaddingSize - 1))
					{
						for (int col = 0; col < outputSquareSizeNoPadding + (2 * outputPaddingSize); col++)
						{
							if (col < outputPaddingSize || col > (outputSquareSizeNoPadding + outputPaddingSize - 1))
							{
								assertEquals(0, oPadding.get(batch, feat, row, col), 0); // padding should be black
							}
						}
					}
				}
			}
		}

		return oPadding;
	}

	@Test
	public void testRandomMaxPoolingBP()
	{

		long seed = 123456789;

		int inputSquareSize = 200;
		int inputFeatures = 1;
		int pollingAreaSquareSize = 6;
		int batchSize = 1;
		int stride = 1;
		int paddingSize = 0;

		testRandomMaxPoolingBP(Runtime.CPU_SEQ, seed, inputSquareSize, inputFeatures, pollingAreaSquareSize, batchSize, stride, paddingSize);
		testRandomMaxPoolingBP(Runtime.OPENCL, seed, inputSquareSize, inputFeatures, pollingAreaSquareSize, batchSize, stride, paddingSize);
	}


	private void testRandomMaxPoolingBP(Runtime runtime, long seed, int inputSquareSize, int inputFeatures, int pollingAreaSquareSize, int batchSize, int stride, int outputPaddingSize)
	{

		Random random = new Random(seed);
		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();

		Subsampling2DConnection connection = cf.subsampling2D(new Layer(), new Layer(), inputSquareSize, inputSquareSize, pollingAreaSquareSize, pollingAreaSquareSize, inputFeatures, stride, stride,
				outputPaddingSize, outputPaddingSize);

		List<Connections> connections = new ArrayList<Connections>();
		connections.add(connection);

		// max pooling
		ValuesProvider vpFF = TensorFactory.tensorProvider(connection, batchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[inputSquareSize * inputSquareSize * inputFeatures * batchSize];
		for (int i = 0; i < src.length; i++)
		{
			src[i] = i + random.nextFloat(); // use increasing number in order to avoid chain rule which is hard to test with random numbers
		}
		System.arraycopy(src, 0, vpFF.get(connection.getInputLayer()).getElements(), vpFF.get(connection.getInputLayer()).getStartIndex(), src.length);

		ConnectionCalculator calc = OperationsFactory.maxPooling2D();

		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(calc instanceof OpenCLMaxPooling2DConnectionCalculator);
		} else
		{
			assertTrue(calc instanceof AparapiMaxPooling2D);
		}

		// do max pooling FF
		calc.calculate(connections, vpFF, connection.getOutputLayer());

		// create BP value provider
		ValuesProvider vpBP = TensorFactory.tensorProvider(connection, batchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		// copy FF output (without padding) to BP output without padding
		int outputSquareSizeNoPadding = connection.getOutputFeatureMapRows();
		TensorFactory.copy(vpFF.<Tensor> get(connection.getOutputLayer(), batchSize, inputFeatures, outputSquareSizeNoPadding, outputSquareSizeNoPadding),
				vpBP.get(connection.getOutputLayer(), batchSize, inputFeatures, outputSquareSizeNoPadding, outputSquareSizeNoPadding));

		BackPropagationConnectionCalculator bp = OperationsFactory.bpMaxPooling(new Properties());

		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(bp instanceof OpenCLMaxPooling2DBP);
		} else
		{
			assertTrue(bp instanceof BackpropagationMaxPooling2D2);
		}

		// set activations
		bp.setActivations(vpFF);
		// do BP max pooling
		bp.calculate(connections, vpBP, connection.getInputLayer());

		Tensor inputFF = vpFF.get(connection.getInputLayer());
		Tensor inputBP = vpBP.get(connection.getInputLayer());
		assertNotNull(inputFF);
		assertNotNull(inputBP);

		for (int batch = 0; batch < batchSize; batch++)
		{
			for (int feat = 0; feat < inputFeatures; feat++)
			{
				for (int row = 0; row < inputSquareSize; row++)
				{
					for (int col = 0; col < inputSquareSize; col++)
					{
						// should be 0 everwhere but at the max areas
						if (inputBP.get(batch, feat, row, col) != 0)
						{
							assertEquals(inputBP.get(batch, feat, row, col), inputFF.get(batch, feat, row, col), 0);
						}
					}
				}
			}
		}
	}

	@Test
	public void testNominalMaxPoolingBFStrideOne()
	{
		testMaxPoolingBPStrideOne(Runtime.CPU_SEQ);
		testMaxPoolingBPStrideOne(Runtime.OPENCL);
	}

	private void testMaxPoolingBPStrideOne(Runtime runtime)
	{

		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();
		// 4x4 feature maps (size of the while input image), 2x2 subsampling area (polling rect), 2 filters (number of input feature maps), 2 row stride, 2 column stride, 2 out row padding, 1 out column
		// padding
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 1, 1, 1, 0, 0);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ConnectionCalculator calc = OperationsFactory.maxPooling2D();

		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(calc instanceof OpenCLMaxPooling2DConnectionCalculator);
		} else
		{
			assertTrue(calc instanceof AparapiMaxPooling2D);
		}

		ValuesProvider vp = TensorFactory.tensorProvider(c, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);
		calc.calculate(connections, vp, c.getOutputLayer());
		Tensor o = vp.get(c.getOutputLayer(), 1, 1, 3, 3); // tensor without padding
		assertNotNull(o);
		// test values
		assertEquals(6, o.get(0, 0, 0, 0), 0);
		assertEquals(7, o.get(0, 0, 0, 1), 0);
		assertEquals(8, o.get(0, 0, 0, 2), 0);
		assertEquals(10, o.get(0, 0, 1, 0), 0);
		assertEquals(11, o.get(0, 0, 1, 1), 0);
		assertEquals(12, o.get(0, 0, 1, 2), 0);
		assertEquals(14, o.get(0, 0, 2, 0), 0);
		assertEquals(15, o.get(0, 0, 2, 1), 0);
		assertEquals(16, o.get(0, 0, 2, 2), 0);


		// create BP value provider
		ValuesProvider vpBP = TensorFactory.tensorProvider(c, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		// copy FF output (without padding) to BP output without padding
		int outputSquareSizeNoPadding = c.getOutputFeatureMapRows();
		TensorFactory.copy(vp.<Tensor> get(c.getOutputLayer(), 1, 1, outputSquareSizeNoPadding, outputSquareSizeNoPadding),
				vpBP.get(c.getOutputLayer(), 1, 1, outputSquareSizeNoPadding, outputSquareSizeNoPadding));

		BackPropagationConnectionCalculator bp = OperationsFactory.bpMaxPooling(new Properties());

		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(bp instanceof OpenCLMaxPooling2DBP);
		} else
		{
			assertTrue(bp instanceof BackpropagationMaxPooling2D2);
		}

		bp.setActivations(vp);
		bp.calculate(connections, vpBP, c.getInputLayer());


		Tensor inputBP = vpBP.get(c.getInputLayer());

		assertEquals(0, inputBP.get(0, 0, 0, 0), 0);
		assertEquals(0, inputBP.get(0, 0, 0, 1), 0);
		assertEquals(0, inputBP.get(0, 0, 0, 2), 0);
		assertEquals(0, inputBP.get(0, 0, 0, 3), 0);
		assertEquals(0, inputBP.get(0, 0, 1, 0), 0);
		assertEquals(6, inputBP.get(0, 0, 1, 1), 0);
		assertEquals(7, inputBP.get(0, 0, 1, 2), 0);
		assertEquals(8, inputBP.get(0, 0, 1, 3), 0);
		assertEquals(0, inputBP.get(0, 0, 2, 0), 0);
		assertEquals(10, inputBP.get(0, 0, 2, 1), 0);
		assertEquals(11, inputBP.get(0, 0, 2, 2), 0);
		assertEquals(12, inputBP.get(0, 0, 2, 3), 0);
		assertEquals(0, inputBP.get(0, 0, 3, 0), 0);
		assertEquals(14, inputBP.get(0, 0, 3, 1), 0);
		assertEquals(15, inputBP.get(0, 0, 3, 2), 0);
		assertEquals(16, inputBP.get(0, 0, 3, 3), 0);

	}


	@Test
	public void testNominalMaxPoolingBPChainRule()
	{
		testMaxPoolingBPChainRule(Runtime.CPU_SEQ);
		testMaxPoolingBPChainRule(Runtime.OPENCL);
	}

	private void testMaxPoolingBPChainRule(Runtime runtime)
	{

		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();
		// 4x4 feature maps (size of the while input image), 2x2 subsampling area (polling rect), 2 filters (number of input feature maps), 2 row stride, 2 column stride, 2 out row padding, 1 out column
		// padding
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 1, 1, 1, 0, 0);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ConnectionCalculator calc = OperationsFactory.maxPooling2D();

		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(calc instanceof OpenCLMaxPooling2DConnectionCalculator);
		} else
		{
			assertTrue(calc instanceof AparapiMaxPooling2D);
		}

		ValuesProvider vp = TensorFactory.tensorProvider(c, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 1, 2, 3, 4, 5, 100, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }; // 100 will be winner in 4 patches, i.e. bp will accumulate to 400
		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);
		calc.calculate(connections, vp, c.getOutputLayer());
		Tensor o = vp.get(c.getOutputLayer(), 1, 1, 3, 3); // tensor without padding
		assertNotNull(o);
		// test values
		assertEquals(100, o.get(0, 0, 0, 0), 0);
		assertEquals(100, o.get(0, 0, 0, 1), 0);
		assertEquals(8, o.get(0, 0, 0, 2), 0);
		assertEquals(100, o.get(0, 0, 1, 0), 0);
		assertEquals(100, o.get(0, 0, 1, 1), 0);
		assertEquals(12, o.get(0, 0, 1, 2), 0);
		assertEquals(14, o.get(0, 0, 2, 0), 0);
		assertEquals(15, o.get(0, 0, 2, 1), 0);
		assertEquals(16, o.get(0, 0, 2, 2), 0);


		// create BP value provider
		ValuesProvider vpBP = TensorFactory.tensorProvider(c, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		// copy FF output (without padding) to BP output without padding
		int outputSquareSizeNoPadding = c.getOutputFeatureMapRows();
		TensorFactory.copy(vp.<Tensor> get(c.getOutputLayer(), 1, 1, outputSquareSizeNoPadding, outputSquareSizeNoPadding),
				vpBP.get(c.getOutputLayer(), 1, 1, outputSquareSizeNoPadding, outputSquareSizeNoPadding));

		BackPropagationConnectionCalculator bp = OperationsFactory.bpMaxPooling(new Properties());

		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(bp instanceof OpenCLMaxPooling2DBP);
		} else
		{
			assertTrue(bp instanceof BackpropagationMaxPooling2D2);
		}

		bp.setActivations(vp);
		bp.calculate(connections, vpBP, c.getInputLayer());


		Tensor inputBP = vpBP.get(c.getInputLayer());

		assertEquals(0, inputBP.get(0, 0, 0, 0), 0);
		assertEquals(0, inputBP.get(0, 0, 0, 1), 0);
		assertEquals(0, inputBP.get(0, 0, 0, 2), 0);
		assertEquals(0, inputBP.get(0, 0, 0, 3), 0);
		assertEquals(0, inputBP.get(0, 0, 1, 0), 0);
		assertEquals(400, inputBP.get(0, 0, 1, 1), 0); // 400 comes from 4 patches where 100 was the maximum
		assertEquals(0, inputBP.get(0, 0, 1, 2), 0);
		assertEquals(8, inputBP.get(0, 0, 1, 3), 0);
		assertEquals(0, inputBP.get(0, 0, 2, 0), 0);
		assertEquals(0, inputBP.get(0, 0, 2, 1), 0);
		assertEquals(0, inputBP.get(0, 0, 2, 2), 0);
		assertEquals(12, inputBP.get(0, 0, 2, 3), 0);
		assertEquals(0, inputBP.get(0, 0, 3, 0), 0);
		assertEquals(14, inputBP.get(0, 0, 3, 1), 0);
		assertEquals(15, inputBP.get(0, 0, 3, 2), 0);
		assertEquals(16, inputBP.get(0, 0, 3, 3), 0);

	}

	@Test
	public void testNominalMaxPoolingFF()
	{
		testMaxPooling(Runtime.CPU_SEQ);
		testMaxPooling(Runtime.OPENCL);
	}

	private void testMaxPooling(Runtime runtime)
	{

		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();
		// 4x4 feature maps (size of the while input image), 2x2 subsampling area (polling rect), 2 filters (number of input feature maps), 2 row stride, 2 column stride, 2 out row padding, 1 out column
		// padding
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 2, 2, 2, 2, 1);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ConnectionCalculator calc = OperationsFactory.maxPooling2D();

		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(calc instanceof OpenCLMaxPooling2DConnectionCalculator);
		} else
		{
			assertTrue(calc instanceof AparapiMaxPooling2D);
		}

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 8.5f, 9f, 9.5f, 10f, 10.5f, 11f, 11.5f, 12f, 12.5f, 13f, 13.5f, 14f, 14.5f, 15f, 16f,
				15.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31 };
		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);

		calc.calculate(connections, vp, c.getOutputLayer());

		// int[] outputDim = new int[] { miniBatchSize, cc.getFilters(), cc.getOutputFeatureMapRows() + 2 * cc.getOutputRowPadding(), cc.getOutputFeatureMapColumns() + 2 * cc.getOutputColumnPadding() };
		Tensor o = vp.get(c.getOutputLayer(), 2, 2, 2, 2); // tensor without padding
		// output size = (4 - (2-2))^2 (features size - (subsampling - stride))^2
		// test padding
		Tensor oPadding = vp.get(c.getOutputLayer(), 2, 2, 6, 4); // tensor with padding
		assertTrue(o.getElements() == oPadding.getElements());

		// test values
		assertEquals(3, o.get(0, 0, 0, 0), 0);
		assertEquals(4, o.get(0, 0, 0, 1), 0);
		assertEquals(7, o.get(0, 0, 1, 0), 0);
		assertEquals(8, o.get(0, 0, 1, 1), 0);
		assertEquals(11, o.get(0, 1, 0, 0), 0);
		assertEquals(12, o.get(0, 1, 0, 1), 0);
		assertEquals(15, o.get(0, 1, 1, 0), 0);
		assertEquals(16, o.get(0, 1, 1, 1), 0);

		assertEquals(6, o.get(1, 0, 0, 0), 0);
		assertEquals(8, o.get(1, 0, 0, 1), 0);
		assertEquals(14, o.get(1, 0, 1, 0), 0);
		assertEquals(16, o.get(1, 0, 1, 1), 0);
		assertEquals(22, o.get(1, 1, 0, 0), 0);
		assertEquals(24, o.get(1, 1, 0, 1), 0);
		assertEquals(30, o.get(1, 1, 1, 0), 0);
		assertEquals(32, o.get(1, 1, 1, 1), 0);
	}

	@Test
	public void testNominalMaxPoolingBP()
	{
		testMaxPoolingBackpropagation(Runtime.CPU_SEQ);
		testMaxPoolingBackpropagation(Runtime.OPENCL);
	}

	private void testMaxPoolingBackpropagation(Runtime runtime)
	{

		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();

		Subsampling2DConnection connection = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 2, 2, 2, 2, 1);

		List<Connections> connections = new ArrayList<Connections>();
		connections.add(connection);

		// max pooling
		ValuesProvider vpFF = TensorFactory.tensorProvider(connection, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 8.5f, 9f, 9.5f, 10f, 10.5f, 11f, 11.5f, 12f, 12.5f, 13f, 13.5f, 14f, 14.5f, 15f, 16f,
				15.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31 };
		System.arraycopy(src, 0, vpFF.get(connection.getInputLayer()).getElements(), vpFF.get(connection.getInputLayer()).getStartIndex(), src.length);

		ConnectionCalculator calc = OperationsFactory.maxPooling2D();

		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(calc instanceof OpenCLMaxPooling2DConnectionCalculator);
		} else
		{
			assertTrue(calc instanceof AparapiMaxPooling2D);
		}

		// do max pooling FF
		calc.calculate(connections, vpFF, connection.getOutputLayer());

		// create BP value provider
		ValuesProvider vpBP = TensorFactory.tensorProvider(connection, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		// copy FF output (without padding) to BP output
		TensorFactory.copy(vpFF.<Tensor> get(connection.getOutputLayer(), 2, 2, 2, 2), vpBP.get(connection.getOutputLayer(), 2, 2, 2, 2));

		BackPropagationConnectionCalculator bp = OperationsFactory.bpMaxPooling(new Properties());

		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(bp instanceof OpenCLMaxPooling2DBP);
		} else
		{
			assertTrue(bp instanceof BackpropagationMaxPooling2D2);
		}

		// set activations
		bp.setActivations(vpFF);
		// do BP max pooling
		bp.calculate(connections, vpBP, connection.getInputLayer());

		Tensor inputFF = vpFF.get(connection.getInputLayer());
		Tensor inputBP = vpBP.get(connection.getInputLayer());

		// input after BP should be the same as the input of the max pooling
		assertEquals(true, inputBP.get(0, 0, 1, 1) == inputFF.get(0, 0, 1, 1));
		assertEquals(true, inputBP.get(0, 0, 1, 3) == inputFF.get(0, 0, 1, 3));
		assertEquals(true, inputBP.get(0, 0, 3, 1) == inputFF.get(0, 0, 3, 1));
		assertEquals(true, inputBP.get(0, 0, 3, 2) == inputFF.get(0, 0, 3, 2));
		assertEquals(true, inputBP.get(0, 1, 1, 1) == inputFF.get(0, 1, 1, 1));
		assertEquals(true, inputBP.get(0, 1, 1, 3) == inputFF.get(0, 1, 1, 3));
		assertEquals(true, inputBP.get(0, 1, 3, 1) == inputFF.get(0, 1, 3, 1));
		assertEquals(true, inputBP.get(0, 1, 3, 2) == inputFF.get(0, 1, 3, 2));

		assertEquals(true, inputBP.get(1, 0, 1, 1) == inputFF.get(1, 0, 1, 1));
		assertEquals(true, inputBP.get(1, 0, 1, 3) == inputFF.get(1, 0, 1, 3));
		assertEquals(true, inputBP.get(1, 0, 3, 1) == inputFF.get(1, 0, 3, 1));
		assertEquals(true, inputBP.get(1, 0, 3, 2) == inputFF.get(1, 0, 3, 2));
		assertEquals(true, inputBP.get(1, 1, 1, 1) == inputFF.get(1, 1, 1, 1));
		assertEquals(true, inputBP.get(1, 1, 1, 3) == inputFF.get(1, 1, 1, 3));
		assertEquals(true, inputBP.get(1, 1, 3, 1) == inputFF.get(1, 1, 3, 1));
		assertEquals(true, inputBP.get(1, 1, 3, 2) == inputFF.get(1, 1, 3, 2));
	}


	@Test
	public void testSubsamplingStride()
	{
		testSubsamplingStride(Runtime.CPU_SEQ);
		testSubsamplingStride(Runtime.OPENCL);
	}

	// / from CNNTest
	private void testSubsamplingStride(Runtime runtime)
	{

		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 2, 2, 1, 0, 0);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ConnectionCalculator calc = OperationsFactory.maxPooling2D();

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 8.5f, 9f, 9.5f, 10f, 10.5f, 11f, 11.5f, 12f, 12.5f, 13f, 13.5f, 14f, 14.5f, 15f, 16f,
				15.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31 };
		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);

		calc.calculate(connections, vp, c.getOutputLayer());

		Tensor o = vp.get(c.getOutputLayer());
		assertEquals(2, o.getDimensions()[2]);
		assertEquals(3, o.getDimensions()[3]);

		assertEquals(3, o.get(0, 0, 0, 0), 0);
		assertEquals(3.5, o.get(0, 0, 0, 1), 0);
		assertEquals(4, o.get(0, 0, 0, 2), 0);
		assertEquals(7, o.get(0, 0, 1, 0), 0);
		assertEquals(8, o.get(0, 0, 1, 1), 0);
		assertEquals(8, o.get(0, 0, 1, 2), 0);

		assertEquals(6, o.get(1, 0, 0, 0), 0);
		assertEquals(7, o.get(1, 0, 0, 1), 0);
		assertEquals(8, o.get(1, 0, 0, 2), 0);
		assertEquals(14, o.get(1, 0, 1, 0), 0);
		assertEquals(16, o.get(1, 0, 1, 1), 0);
		assertEquals(16, o.get(1, 0, 1, 2), 0);

		assertEquals(11, o.get(0, 1, 0, 0), 0);
		assertEquals(11.5, o.get(0, 1, 0, 1), 0);
		assertEquals(12, o.get(0, 1, 0, 2), 0);
		assertEquals(15, o.get(0, 1, 1, 0), 0);
		assertEquals(16, o.get(0, 1, 1, 1), 0);
		assertEquals(16, o.get(0, 1, 1, 2), 0);

		assertEquals(22, o.get(1, 1, 0, 0), 0);
		assertEquals(23, o.get(1, 1, 0, 1), 0);
		assertEquals(24, o.get(1, 1, 0, 2), 0);
		assertEquals(30, o.get(1, 1, 1, 0), 0);
		assertEquals(32, o.get(1, 1, 1, 1), 0);
		assertEquals(32, o.get(1, 1, 1, 2), 0);
	}

	@Test
	public void testSubsamplingStrideBackpropagation()
	{
		testSubsamplingStrideBackpropagation(Runtime.CPU_SEQ);
		testSubsamplingStrideBackpropagation(Runtime.OPENCL);
	}

	// / from CNNTest
	private void testSubsamplingStrideBackpropagation(Runtime runtime)
	{

		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 2, 2, 1, 0, 0);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		// max pooling
		ValuesProvider activations = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 8.5f, 9f, 9.5f, 10f, 10.5f, 11f, 11.5f, 12f, 12.5f, 13f, 13.5f, 14f, 14.5f, 15f, 16f,
				15.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31 };
		System.arraycopy(src, 0, activations.get(c.getInputLayer()).getElements(), activations.get(c.getInputLayer()).getStartIndex(), src.length);

		ConnectionCalculator calc = OperationsFactory.maxPooling2D();
		calc.calculate(connections, activations, c.getOutputLayer());

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		TensorFactory.copy(activations.<Tensor> get(c.getOutputLayer()), vp.get(c.getOutputLayer()));

		BackPropagationConnectionCalculator bp = OperationsFactory.bpMaxPooling(new Properties());
		bp.setActivations(activations);
		bp.calculate(connections, vp, c.getInputLayer());

		Tensor a = activations.get(c.getInputLayer());
		Tensor bpo = vp.get(c.getInputLayer());

		assertEquals(false, bpo.get(0, 0, 0, 1) >= a.get(0, 0, 0, 1));
		assertEquals(true, bpo.get(0, 0, 1, 1) >= a.get(0, 0, 1, 1));
		assertEquals(true, bpo.get(0, 0, 1, 2) >= a.get(0, 0, 1, 2));
		assertEquals(false, bpo.get(0, 0, 3, 3) >= a.get(0, 0, 3, 3));
		assertEquals(true, bpo.get(0, 0, 3, 1) >= a.get(0, 0, 3, 1));
		assertEquals(true, bpo.get(0, 0, 3, 2) >= a.get(0, 0, 3, 2));

		assertEquals(false, bpo.get(0, 1, 0, 1) >= a.get(0, 1, 0, 1));
		assertEquals(true, bpo.get(0, 1, 1, 1) >= a.get(0, 1, 1, 1));
		assertEquals(true, bpo.get(0, 1, 1, 2) >= a.get(0, 1, 1, 2));
		assertEquals(false, bpo.get(0, 1, 3, 3) >= a.get(0, 1, 3, 3));
		assertEquals(true, bpo.get(0, 1, 3, 1) >= a.get(0, 1, 3, 1));
		assertEquals(true, bpo.get(0, 1, 3, 2) >= a.get(0, 1, 3, 2));

		assertEquals(false, bpo.get(1, 0, 0, 1) >= a.get(1, 0, 0, 1));
		assertEquals(true, bpo.get(1, 0, 1, 1) >= a.get(1, 0, 1, 1));
		assertEquals(true, bpo.get(1, 0, 1, 2) >= a.get(1, 0, 1, 2));
		assertEquals(false, bpo.get(1, 0, 3, 3) >= a.get(1, 0, 3, 3));
		assertEquals(true, bpo.get(1, 0, 3, 1) >= a.get(1, 0, 3, 1));
		assertEquals(true, bpo.get(1, 0, 3, 2) >= a.get(1, 0, 3, 2));

		assertEquals(false, bpo.get(1, 1, 0, 1) >= a.get(1, 1, 0, 1));
		assertEquals(true, bpo.get(1, 1, 1, 1) >= a.get(1, 1, 1, 1));
		assertEquals(true, bpo.get(1, 1, 1, 2) >= a.get(1, 1, 1, 2));
		assertEquals(false, bpo.get(1, 1, 3, 3) >= a.get(1, 1, 3, 3));
		assertEquals(true, bpo.get(1, 1, 3, 1) >= a.get(1, 1, 3, 1));
		assertEquals(true, bpo.get(1, 1, 3, 2) >= a.get(1, 1, 3, 2));
	}

	@Test
	public void testMaxPoolingOverlapping()
	{
		testMaxPoolingOverlapping(Runtime.CPU_SEQ);
		testMaxPoolingOverlapping(Runtime.OPENCL);
	}

	private void testMaxPoolingOverlapping(Runtime runtime)
	{
		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 6, 6, 3, 3, 1, 2, 2, 0, 0);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		// feedforward
		ConnectionCalculator calc = OperationsFactory.maxPooling2D();

		ValuesProvider activations = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 29, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 15, 30, 31, 32, 33, 34, 35, 36,
																1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 29, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 15, 30, 31, 32, 33, 34, 35, 36 };
		System.arraycopy(src, 0, activations.get(c.getInputLayer()).getElements(), activations.get(c.getInputLayer()).getStartIndex(), src.length);

		calc.calculate(connections, activations, c.getOutputLayer());

		Tensor o = activations.get(c.getOutputLayer());

		for (int i = 0; i < 2; i++)
		{
			assertEquals(29, o.get(i, 0, 0, 0), 0);
			assertEquals(29, o.get(i, 0, 0, 1), 0);
			assertEquals(29, o.get(i, 0, 1, 0), 0);
			assertEquals(29, o.get(i, 0, 1, 1), 0);
		}

		// backpropagation
		BackPropagationConnectionCalculator bp = OperationsFactory.bpMaxPooling(new Properties());
		bp.setActivations(activations);

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		TensorFactory.copy(activations.<Tensor> get(c.getOutputLayer()), vp.get(c.getOutputLayer()));

		bp.calculate(connections, vp, c.getInputLayer());

		Tensor bpo = vp.get(c.getInputLayer());

		for (int i = 0; i < 2; i++)
		{
			assertEquals(0, bpo.get(i, 0, 0, 0), 0);
			assertEquals(0, bpo.get(i, 0, 0, 1), 0);
			assertEquals(0, bpo.get(i, 0, 0, 2), 0);
			assertEquals(0, bpo.get(i, 0, 0, 3), 0);
			assertEquals(0, bpo.get(i, 0, 0, 4), 0);
			assertEquals(0, bpo.get(i, 0, 0, 5), 0);
			
			assertEquals(0, bpo.get(i, 0, 1, 0), 0);
			assertEquals(0, bpo.get(i, 0, 1, 1), 0);
			assertEquals(0, bpo.get(i, 0, 1, 2), 0);
			assertEquals(0, bpo.get(i, 0, 1, 3), 0);
			assertEquals(0, bpo.get(i, 0, 1, 4), 0);
			assertEquals(0, bpo.get(i, 0, 1, 5), 0);
			
			assertEquals(0, bpo.get(i, 0, 2, 0), 0);
			assertEquals(0, bpo.get(i, 0, 2, 1), 0);
			assertEquals(29 * 4, bpo.get(i, 0, 2, 2), 0.000001);
			assertEquals(0, bpo.get(i, 0, 2, 3), 0);
			assertEquals(0, bpo.get(i, 0, 2, 4), 0);
			assertEquals(0, bpo.get(i, 0, 2, 5), 0);
			
			assertEquals(0, bpo.get(i, 0, 3, 0), 0);
			assertEquals(0, bpo.get(i, 0, 3, 1), 0);
			assertEquals(0, bpo.get(i, 0, 3, 2), 0.000001);
			assertEquals(0, bpo.get(i, 0, 3, 3), 0);
			assertEquals(0, bpo.get(i, 0, 3, 4), 0);
			assertEquals(0, bpo.get(i, 0, 3, 5), 0);
			
			assertEquals(0, bpo.get(i, 0, 4, 0), 0);
			assertEquals(0, bpo.get(i, 0, 4, 1), 0);
			assertEquals(0, bpo.get(i, 0, 4, 2), 0.000001);
			assertEquals(0, bpo.get(i, 0, 4, 3), 0);
			assertEquals(0, bpo.get(i, 0, 4, 4), 0);
			assertEquals(0, bpo.get(i, 0, 4, 5), 0);
			
			assertEquals(0, bpo.get(i, 0, 5, 0), 0);
			assertEquals(0, bpo.get(i, 0, 5, 1), 0);
			assertEquals(0, bpo.get(i, 0, 5, 2), 0);
			assertEquals(0, bpo.get(i, 0, 5, 3), 0);
			assertEquals(0, bpo.get(i, 0, 5, 4), 0);
			assertEquals(0, bpo.get(i, 0, 5, 5), 0);
		}
	}

}
