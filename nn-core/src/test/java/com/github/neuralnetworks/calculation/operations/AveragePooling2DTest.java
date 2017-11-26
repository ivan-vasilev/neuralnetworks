package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertEquals;
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
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiAveragePooling2D;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.BackpropagationAveragePooling2D2;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLAveragePooling2DBP;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLAveragePooling2DConnectionCalculator;
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
public class AveragePooling2DTest extends AbstractTest
{

	@Test
	public void testRandomAveragePoolingFF()
	{
		long seed = 123456789;

		int inputSquareSize = 200;
		int inputFeatures = 3;
		int pollingAreaSquareSize = 5;
		int batchSize = 2;
		int stride = 1;
		int paddingSize = 0;


		Tensor seqResult = testRandomAveragePoolingFF(Runtime.CPU_SEQ, seed, inputSquareSize, inputFeatures, pollingAreaSquareSize, batchSize, stride, paddingSize);
		Tensor openclResult = testRandomAveragePoolingFF(Runtime.OPENCL, seed, inputSquareSize, inputFeatures, pollingAreaSquareSize, batchSize, stride, paddingSize);

		assertTrue(isEqual(seqResult, openclResult));

	}

	// from CNNTests
	private Tensor testRandomAveragePoolingFF(Runtime runtime, long seed, int inputSquareSize, int inputFeatures, int pollingAreaSquareSize, int batchSize, int stride, int paddingSize)
	{

		Random random = new Random(seed);

		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), inputSquareSize, inputSquareSize, pollingAreaSquareSize, pollingAreaSquareSize, inputFeatures, stride, stride, paddingSize,
				paddingSize);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ConnectionCalculator calc = OperationsFactory.averagePooling2D();

		ValuesProvider vp = TensorFactory.tensorProvider(c, batchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[batchSize * inputFeatures * inputSquareSize * inputSquareSize];
		for (int i = 0; i < src.length; i++)
		{
			src[i] = random.nextFloat();
		}


		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);

		calc.calculate(connections, vp, c.getOutputLayer());

		Tensor o = vp.get(c.getOutputLayer());

		return o;
	}

	@Test
	public void testAveragePoolingBackpropagationOverlapping()
	{
		testAveragePoolingBackpropagationOverlapping(Runtime.CPU_SEQ);
		testAveragePoolingBackpropagationOverlapping(Runtime.OPENCL);
	}

	private void testAveragePoolingBackpropagationOverlapping(Runtime runtime)
	{

		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 1, 1, 1, 0, 0);

		// average pooling
		ConnectionCalculator calc = OperationsFactory.averagePooling2D();

		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ValuesProvider activations = TensorFactory.tensorProvider(c, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[16];
		for (int i = 0; i < src.length; i++)
		{
			src[i] = i + 1;
		}
		System.arraycopy(src, 0, activations.get(c.getInputLayer()).getElements(), activations.get(c.getInputLayer()).getStartIndex(), src.length);

		calc.calculate(connections, activations, c.getOutputLayer());

		Tensor ffOut = activations.get(c.getOutputLayer());

		assertEquals(3.5f, ffOut.get(0, 0, 0, 0), 0);
		assertEquals(4.5f, ffOut.get(0, 0, 0, 1), 0);
		assertEquals(5.5f, ffOut.get(0, 0, 0, 2), 0);
		assertEquals(7.5f, ffOut.get(0, 0, 1, 0), 0);
		assertEquals(8.5f, ffOut.get(0, 0, 1, 1), 0);
		assertEquals(9.5f, ffOut.get(0, 0, 1, 2), 0);
		assertEquals(11.5f, ffOut.get(0, 0, 2, 0), 0);
		assertEquals(12.5f, ffOut.get(0, 0, 2, 1), 0);
		assertEquals(13.5f, ffOut.get(0, 0, 2, 2), 0);


		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(calc instanceof OpenCLAveragePooling2DConnectionCalculator);
		} else
		{
			assertTrue(calc instanceof AparapiAveragePooling2D);
		}

		BackPropagationConnectionCalculator bp = OperationsFactory.bpAveragePooling(new Properties());

		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(bp instanceof OpenCLAveragePooling2DBP);
		} else
		{
			assertTrue(bp instanceof BackpropagationAveragePooling2D2);
		}

		bp.setActivations(activations);

		ValuesProvider vp = TensorFactory.tensorProvider(c, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		TensorFactory.copy(activations.<Tensor> get(c.getOutputLayer()), vp.get(c.getOutputLayer()));

		bp.calculate(connections, vp, c.getInputLayer());

		Tensor bpo = vp.get(c.getInputLayer());

		assertEquals(0.875f, bpo.get(0, 0, 0, 0), 0); // 3.5/4
		assertEquals(2f, bpo.get(0, 0, 0, 1), 0); // (3.5+4.5)/4
		assertEquals(2.5f, bpo.get(0, 0, 0, 2), 0); // (4.5+5.5)/4
		assertEquals(1.375f, bpo.get(0, 0, 0, 3), 0); // 5.5/4
		assertEquals(2.75f, bpo.get(0, 0, 1, 0), 0); // (3,5+7,5)/4
		assertEquals(6f, bpo.get(0, 0, 1, 1), 0); // (3.5+4.5+7.5+8.5)/4
		assertEquals(7f, bpo.get(0, 0, 1, 2), 0); // (4,5+5,5+8,5+9,5)/4
		assertEquals(3.75f, bpo.get(0, 0, 1, 3), 0); // (5,5+9,5)/4
		assertEquals(4.75f, bpo.get(0, 0, 2, 0), 0); // (7,5+11,5)/4
		assertEquals(10f, bpo.get(0, 0, 2, 1), 0); // (7,5+8,5+11,5+12,5)/4
		assertEquals(11f, bpo.get(0, 0, 2, 2), 0); // (8,5+9,5+12,5+13,5)/4
		assertEquals(5.75f, bpo.get(0, 0, 2, 3), 0); // (9,5+13,5)/4
		assertEquals(2.875f, bpo.get(0, 0, 3, 0), 0); // 11,5/4
		assertEquals(6f, bpo.get(0, 0, 3, 1), 0); // (11,5+12,5)/4
		assertEquals(6.5f, bpo.get(0, 0, 3, 2), 0); // (12,5+13,5)/4
		assertEquals(3.375f, bpo.get(0, 0, 3, 3), 0); // 13,5/4
	}

	@Test
	public void testRandomAveragePoolingBP()
	{
		long seed = 33456386;

		int inputSquareSize = 200;
		int batchSize = 5;
		int poolingAreaSquareSize = 5; // has to be a divider of inputSqareSize
		int features = 3;

		Tensor seqResult = testRandomAveragePoolingBP(Runtime.CPU_SEQ, seed, inputSquareSize, batchSize, poolingAreaSquareSize, features);
		Tensor openclResult = testRandomAveragePoolingBP(Runtime.OPENCL, seed, inputSquareSize, batchSize, poolingAreaSquareSize, features);

		assertTrue(isEqual(seqResult, openclResult));
	}

	private Tensor testRandomAveragePoolingBP(Runtime runtime, long seed, int inputSquareSize, int batchSize, int poolingAreaSquareSize, int features)
	{

		Random random = new Random(seed);
		int stride = poolingAreaSquareSize; // use non overlapping stride

		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), inputSquareSize, inputSquareSize, poolingAreaSquareSize, poolingAreaSquareSize, features, stride, stride, 0, 0);

		// average pooling
		ConnectionCalculator calc = OperationsFactory.averagePooling2D();

		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ValuesProvider activations = TensorFactory.tensorProvider(c, batchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[batchSize * features * inputSquareSize * inputSquareSize];
		for (int i = 0; i < src.length; i++)
		{
			src[i] = random.nextFloat();
		}
		System.arraycopy(src, 0, activations.get(c.getInputLayer()).getElements(), activations.get(c.getInputLayer()).getStartIndex(), src.length);
		calc.calculate(connections, activations, c.getOutputLayer());

		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(calc instanceof OpenCLAveragePooling2DConnectionCalculator);
		} else
		{
			assertTrue(calc instanceof AparapiAveragePooling2D);
		}

		BackPropagationConnectionCalculator bp = OperationsFactory.bpAveragePooling(new Properties());

		if (runtime.equals(Runtime.OPENCL))
		{
			assertTrue(bp instanceof OpenCLAveragePooling2DBP);
		} else
		{
			assertTrue(bp instanceof BackpropagationAveragePooling2D2);
		}

		bp.setActivations(activations);

		ValuesProvider vp = TensorFactory.tensorProvider(c, batchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		TensorFactory.copy(activations.<Tensor> get(c.getOutputLayer()), vp.get(c.getOutputLayer()));

		bp.calculate(connections, vp, c.getInputLayer());

		Tensor bpo = vp.get(c.getInputLayer());

		return bpo;
	}

	@Test
	public void testAveragePooling()
	{
		testAveragePooling(Runtime.CPU_SEQ);
		testAveragePooling(Runtime.OPENCL);
	}

	// from CNNTests
	private void testAveragePooling(Runtime runtime)
	{

		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 2, 2, 2, 0, 0);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ConnectionCalculator calc = OperationsFactory.averagePooling2D();

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 8.5f, 9f, 9.5f, 10f, 10.5f, 11f, 11.5f, 12f, 12.5f, 13f, 13.5f, 14f, 14.5f, 15f, 16f,
				15.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31 };
		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);

		calc.calculate(connections, vp, c.getOutputLayer());

		Tensor o = vp.get(c.getOutputLayer());

		assertEquals(1.75, o.get(0, 0, 0, 0), 0);
		assertEquals(2.75, o.get(0, 0, 0, 1), 0);
		assertEquals(5.75, o.get(0, 0, 1, 0), 0);
		assertEquals(6.75, o.get(0, 0, 1, 1), 0);
		assertEquals(9.75, o.get(0, 1, 0, 0), 0);
		assertEquals(10.75, o.get(0, 1, 0, 1), 0);
		assertEquals(13.75, o.get(0, 1, 1, 0), 0);
		assertEquals(14.75, o.get(0, 1, 1, 1), 0);

		assertEquals(3.5, o.get(1, 0, 0, 0), 0);
		assertEquals(5.5, o.get(1, 0, 0, 1), 0);
		assertEquals(11.5, o.get(1, 0, 1, 0), 0);
		assertEquals(13.5, o.get(1, 0, 1, 1), 0);
		assertEquals(19.5, o.get(1, 1, 0, 0), 0);
		assertEquals(21.5, o.get(1, 1, 0, 1), 0);
		assertEquals(27.5, o.get(1, 1, 1, 0), 0);
		assertEquals(29.5, o.get(1, 1, 1, 1), 0);
	}

	@Test
	public void testAveragePoolingBackpropagation()
	{
		testAveragePoolingBackpropagation(Runtime.CPU_SEQ);
		testAveragePoolingBackpropagation(Runtime.OPENCL);
	}

	// from CNNTests
	private void testAveragePoolingBackpropagation(Runtime runtime)
	{

		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 2, 2, 2, 0, 0);

		// average pooling
		ConnectionCalculator calc = OperationsFactory.averagePooling2D();

		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ValuesProvider activations = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 8.5f, 9f, 9.5f, 10f, 10.5f, 11f, 11.5f, 12f, 12.5f, 13f, 13.5f, 14f, 14.5f, 15f, 16f,
				15.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31 };
		System.arraycopy(src, 0, activations.get(c.getInputLayer()).getElements(), activations.get(c.getInputLayer()).getStartIndex(), src.length);

		calc.calculate(connections, activations, c.getOutputLayer());

		BackPropagationConnectionCalculator bp = OperationsFactory.bpAveragePooling(new Properties());
		bp.setActivations(activations);

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		TensorFactory.copy(activations.<Tensor> get(c.getOutputLayer()), vp.get(c.getOutputLayer()));

		bp.calculate(connections, vp, c.getInputLayer());

		Tensor o = activations.get(c.getOutputLayer());
		Tensor bpo = vp.get(c.getInputLayer());
		assertEquals(o.get(0, 0, 0, 0) / c.getSubsamplingRegionLength(), bpo.get(0, 0, 0, 0), 0);
		assertEquals(o.get(0, 0, 0, 1) / c.getSubsamplingRegionLength(), bpo.get(0, 0, 0, 2), 0);
		assertEquals(o.get(0, 0, 1, 0) / c.getSubsamplingRegionLength(), bpo.get(0, 0, 2, 0), 0);
		assertEquals(o.get(0, 0, 1, 1) / c.getSubsamplingRegionLength(), bpo.get(0, 0, 2, 2), 0);
		assertEquals(o.get(0, 1, 0, 0) / c.getSubsamplingRegionLength(), bpo.get(0, 1, 0, 0), 0);
		assertEquals(o.get(0, 1, 0, 1) / c.getSubsamplingRegionLength(), bpo.get(0, 1, 0, 2), 0);
		assertEquals(o.get(0, 1, 1, 0) / c.getSubsamplingRegionLength(), bpo.get(0, 1, 2, 0), 0);
		assertEquals(o.get(0, 1, 1, 1) / c.getSubsamplingRegionLength(), bpo.get(0, 1, 2, 2), 0);
		assertEquals(o.get(1, 0, 0, 0) / c.getSubsamplingRegionLength(), bpo.get(1, 0, 0, 0), 0);
		assertEquals(o.get(1, 0, 0, 1) / c.getSubsamplingRegionLength(), bpo.get(1, 0, 0, 2), 0);
		assertEquals(o.get(1, 0, 1, 0) / c.getSubsamplingRegionLength(), bpo.get(1, 0, 2, 0), 0);
		assertEquals(o.get(1, 0, 1, 1) / c.getSubsamplingRegionLength(), bpo.get(1, 0, 2, 2), 0);
		assertEquals(o.get(1, 1, 0, 0) / c.getSubsamplingRegionLength(), bpo.get(1, 1, 0, 0), 0);
		assertEquals(o.get(1, 1, 0, 1) / c.getSubsamplingRegionLength(), bpo.get(1, 1, 0, 2), 0);
		assertEquals(o.get(1, 1, 1, 0) / c.getSubsamplingRegionLength(), bpo.get(1, 1, 2, 0), 0);
		assertEquals(o.get(1, 1, 1, 1) / c.getSubsamplingRegionLength(), bpo.get(1, 1, 2, 2), 0);
	}

	@Test
	public void testAveragePoolingOverlapping()
	{
		testAveragePoolingOverlapping(Runtime.CPU_SEQ);
		testAveragePoolingOverlapping(Runtime.OPENCL);
	}

	private void testAveragePoolingOverlapping(Runtime runtime)
	{
		configureGlobalRuntimeEnvironment(runtime);

		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 6, 6, 3, 3, 1, 2, 2, 0, 0);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		// feedforward
		ConnectionCalculator calc = OperationsFactory.averagePooling2D();

		ValuesProvider activations = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
																1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
		System.arraycopy(src, 0, activations.get(c.getInputLayer()).getElements(), activations.get(c.getInputLayer()).getStartIndex(), src.length);

		calc.calculate(connections, activations, c.getOutputLayer());

		Tensor o = activations.get(c.getOutputLayer());

		for (int i = 0; i < 2; i++)
		{
			assertEquals(8, o.get(i, 0, 0, 0), 0);
			assertEquals(10, o.get(i, 0, 0, 1), 0);
			assertEquals(20, o.get(i, 0, 1, 0), 0);
			assertEquals(22, o.get(i, 0, 1, 1), 0);
		}

		// backpropagation
		BackPropagationConnectionCalculator bp = OperationsFactory.bpAveragePooling(new Properties());
		bp.setActivations(activations);

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		TensorFactory.copy(activations.<Tensor> get(c.getOutputLayer()), vp.get(c.getOutputLayer()));

		bp.calculate(connections, vp, c.getInputLayer());

		Tensor bpo = vp.get(c.getInputLayer());

		for (int i = 0; i < 2; i++)
		{
			assertEquals(8 / 9.0f, bpo.get(i, 0, 0, 0), 0);
			assertEquals(8 / 9.0f, bpo.get(i, 0, 0, 1), 0);
			assertEquals(8 / 9.0f + 10 / 9f, bpo.get(i, 0, 0, 2), 0);
			assertEquals(10 / 9f, bpo.get(i, 0, 0, 3), 0);
			assertEquals(10 / 9f, bpo.get(i, 0, 0, 4), 0);
			assertEquals(0, bpo.get(i, 0, 0, 5), 0);
			
			assertEquals(8 / 9.0f, bpo.get(i, 0, 1, 0), 0);
			assertEquals(8 / 9.0f, bpo.get(i, 0, 1, 1), 0);
			assertEquals(8 / 9.0f + 10 / 9f, bpo.get(i, 0, 1, 2), 0);
			assertEquals(10 / 9f, bpo.get(i, 0, 1, 3), 0);
			assertEquals(10 / 9f, bpo.get(i, 0, 1, 4), 0);
			assertEquals(0, bpo.get(i, 0, 1, 5), 0);
			
			assertEquals(8 / 9.0f + 20 / 9f, bpo.get(i, 0, 2, 0), 0);
			assertEquals(8 / 9.0f + 20 / 9f, bpo.get(i, 0, 2, 1), 0);
			assertEquals((8 + 10 + 20 + 22) / 9f, bpo.get(i, 0, 2, 2), 0.000001);
			assertEquals(10 / 9f + 22 / 9f, bpo.get(i, 0, 2, 3), 0);
			assertEquals(10 / 9f + 22 / 9f, bpo.get(i, 0, 2, 4), 0);
			assertEquals(0, bpo.get(i, 0, 2, 5), 0);
			
			assertEquals(20 / 9f, bpo.get(i, 0, 3, 0), 0);
			assertEquals(20 / 9f, bpo.get(i, 0, 3, 1), 0);
			assertEquals((20 + 22) / 9f, bpo.get(i, 0, 3, 2), 0.000001);
			assertEquals(22 / 9f, bpo.get(i, 0, 3, 3), 0);
			assertEquals(22 / 9f, bpo.get(i, 0, 3, 4), 0);
			assertEquals(0, bpo.get(i, 0, 3, 5), 0);
			
			assertEquals(20 / 9f, bpo.get(i, 0, 4, 0), 0);
			assertEquals(20 / 9f, bpo.get(i, 0, 4, 1), 0);
			assertEquals((20 + 22) / 9f, bpo.get(i, 0, 4, 2), 0.000001);
			assertEquals(22 / 9f, bpo.get(i, 0, 4, 3), 0);
			assertEquals(22 / 9f, bpo.get(i, 0, 4, 4), 0);
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
