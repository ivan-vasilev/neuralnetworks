package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.util.Arrays;

import org.junit.Test;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.RepeaterConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.aparapi.LRN;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLLRN;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;

/**
 * Created by chass on 25.11.14.
 *
 * Layer-recurrent network (LRN)
 */
public class LRNTest extends AbstractTest
{

	@Test
	public void testBase()
	{
		ValuesProvider vp = new ValuesProvider(Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		RepeaterConnection c = new RepeaterConnection(new Layer(), new Layer(), 20);
		vp.add(c.getInputLayer(), 2, 5, 2, 2);
		vp.add(c.getOutputLayer(), 2, 5, 2, 2);

		Tensor t = vp.get(c.getInputLayer());
		// 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
		t.set(1, 0, 0, 0, 0);
		t.set(2, 0, 0, 0, 1);
		t.set(3, 0, 0, 1, 0);
		t.set(4, 0, 0, 1, 1);
		t.set(5, 0, 1, 0, 0);
		t.set(6, 0, 1, 0, 1);
		t.set(7, 0, 1, 1, 0);
		t.set(8, 0, 1, 1, 1);
		t.set(9, 0, 2, 0, 0);
		t.set(10, 0, 2, 0, 1);
		t.set(11, 0, 2, 1, 0);
		t.set(12, 0, 2, 1, 1);
		t.set(13, 0, 3, 0, 0);
		t.set(14, 0, 3, 0, 1);
		t.set(15, 0, 3, 1, 0);
		t.set(16, 0, 3, 1, 1);
		t.set(17, 0, 4, 0, 0);
		t.set(18, 0, 4, 0, 1);
		t.set(19, 0, 4, 1, 0);
		t.set(20, 0, 4, 1, 1);

		// 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
		t.set(1, 1, 0, 0, 0);
		t.set(2, 1, 0, 0, 1);
		t.set(3, 1, 0, 1, 0);
		t.set(4, 1, 0, 1, 1);
		t.set(5, 1, 1, 0, 0);
		t.set(6, 1, 1, 0, 1);
		t.set(7, 1, 1, 1, 0);
		t.set(8, 1, 1, 1, 1);
		t.set(9, 1, 2, 0, 0);
		t.set(10, 1, 2, 0, 1);
		t.set(11, 1, 2, 1, 0);
		t.set(12, 1, 2, 1, 1);
		t.set(13, 1, 3, 0, 0);
		t.set(14, 1, 3, 0, 1);
		t.set(15, 1, 3, 1, 0);
		t.set(16, 1, 3, 1, 1);
		t.set(17, 1, 4, 0, 0);
		t.set(18, 1, 4, 0, 1);
		t.set(19, 1, 4, 1, 0);
		t.set(20, 1, 4, 1, 1);

		configureGlobalRuntimeEnvironment(Runtime.OPENCL);

		ConnectionCalculator cc = OperationsFactory.lrnConnectionCalculator(2, 5, 0.01f, 1f);
		cc.calculate(Arrays.asList(new Connections[] { c }), vp, c.getOutputLayer());

		if (cc instanceof OpenCLLRN)
		{
			String kernelOptions = ((OpenCLLRN) cc).kernelOptions(1);
			System.out.println("OpenCL : " + kernelOptions);

			LRN.LRNKernel lrnKernel = ((OpenCLLRN) cc).getAparapi();
			assertNotNull(lrnKernel);

			((OpenCLLRN) cc).destroyKernel();
		}
	}

	@Test
	public void testLRN()
	{
		boolean runOpenCL = true;
		boolean runAparapi = true;

		// FF phase
		ValuesProvider vp = new ValuesProvider(Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		RepeaterConnection c = new RepeaterConnection(new Layer(), new Layer(), 20);
		vp.add(c.getInputLayer(), 2, 5, 2, 2);
		vp.add(c.getOutputLayer(), 2, 5, 2, 2);

		Tensor t = vp.get(c.getInputLayer());
		// 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
		t.set(1, 0, 0, 0, 0);
		t.set(2, 0, 0, 0, 1);
		t.set(3, 0, 0, 1, 0);
		t.set(4, 0, 0, 1, 1);
		t.set(5, 0, 1, 0, 0);
		t.set(6, 0, 1, 0, 1);
		t.set(7, 0, 1, 1, 0);
		t.set(8, 0, 1, 1, 1);
		t.set(9, 0, 2, 0, 0);
		t.set(10, 0, 2, 0, 1);
		t.set(11, 0, 2, 1, 0);
		t.set(12, 0, 2, 1, 1);
		t.set(13, 0, 3, 0, 0);
		t.set(14, 0, 3, 0, 1);
		t.set(15, 0, 3, 1, 0);
		t.set(16, 0, 3, 1, 1);
		t.set(17, 0, 4, 0, 0);
		t.set(18, 0, 4, 0, 1);
		t.set(19, 0, 4, 1, 0);
		t.set(20, 0, 4, 1, 1);

		// 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
		t.set(1, 1, 0, 0, 0);
		t.set(2, 1, 0, 0, 1);
		t.set(3, 1, 0, 1, 0);
		t.set(4, 1, 0, 1, 1);
		t.set(5, 1, 1, 0, 0);
		t.set(6, 1, 1, 0, 1);
		t.set(7, 1, 1, 1, 0);
		t.set(8, 1, 1, 1, 1);
		t.set(9, 1, 2, 0, 0);
		t.set(10, 1, 2, 0, 1);
		t.set(11, 1, 2, 1, 0);
		t.set(12, 1, 2, 1, 1);
		t.set(13, 1, 3, 0, 0);
		t.set(14, 1, 3, 0, 1);
		t.set(15, 1, 3, 1, 0);
		t.set(16, 1, 3, 1, 1);
		t.set(17, 1, 4, 0, 0);
		t.set(18, 1, 4, 0, 1);
		t.set(19, 1, 4, 1, 0);
		t.set(20, 1, 4, 1, 1);

		// OpenCL
		Tensor ffOCLOutput = null;
		Tensor bpOCLOutput = null;
		if (runOpenCL)
		{
			configureGlobalRuntimeEnvironment(Runtime.OPENCL);

			ConnectionCalculator cc = OperationsFactory.lrnConnectionCalculator(2, 5, 0.01f, 1f);

			// perform "cycles" with the opencl calculator
			cc.calculate(Arrays.asList(new Connections[] { c }), vp, c.getOutputLayer());

			ffOCLOutput = TensorFactory.tensor(vp.get(c.getOutputLayer()).getDimensions());
			TensorFactory.copy(vp.get(c.getOutputLayer()), ffOCLOutput);

			assertEquals(0.791557, ffOCLOutput.get(0, 0, 1, 0), 0.0001);
			assertEquals(1.119403, ffOCLOutput.get(0, 1, 0, 1), 0.0001);
			assertEquals(1.176471, ffOCLOutput.get(0, 2, 0, 0), 0.0001);
			assertEquals(2.300406, ffOCLOutput.get(0, 4, 0, 0), 0.0001);
			assertEquals(0.791557, ffOCLOutput.get(1, 0, 1, 0), 0.0001);
			assertEquals(1.119403, ffOCLOutput.get(1, 1, 0, 1), 0.0001);
			assertEquals(1.176471, ffOCLOutput.get(1, 2, 0, 0), 0.0001);
			assertEquals(2.300406, ffOCLOutput.get(1, 4, 0, 0), 0.0001);

			// BP phase
			BackPropagationConnectionCalculator bpLRN = OperationsFactory.bpLRN(new Properties(), cc);
			ValuesProvider bpvp = new ValuesProvider(Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
			bpvp.add(c.getInputLayer(), 2, 5, 2, 2);
			bpvp.add(c.getOutputLayer(), ffOCLOutput);

			bpLRN.setActivations(vp);
			bpLRN.calculate(Arrays.asList(new Connections[] { c }), bpvp, c.getInputLayer());

			bpOCLOutput = bpvp.get(c.getInputLayer());
		}

		// Aparapi
		Tensor ffAparapiOutput = null;
		Tensor bpAparapiOutput = null;
		if (runAparapi)
		{
			configureGlobalRuntimeEnvironment(Runtime.CPU_SEQ);

			ConnectionCalculator cc = OperationsFactory.lrnConnectionCalculator(2, 5, 0.01f, 1f);

			// measure time
			cc.calculate(Arrays.asList(new Connections[] { c }), vp, c.getOutputLayer());

			ffAparapiOutput = TensorFactory.tensor(vp.get(c.getOutputLayer()).getDimensions());
			TensorFactory.copy(vp.get(c.getOutputLayer()), ffAparapiOutput);

			// BP phase
			BackPropagationConnectionCalculator bpLRN = OperationsFactory.bpLRN(new Properties(), cc);
			ValuesProvider bpvp = new ValuesProvider(Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
			bpvp.add(c.getInputLayer(), 2, 5, 2, 2);
			bpvp.add(c.getOutputLayer(), ffAparapiOutput);

			bpLRN.setActivations(vp);
			bpLRN.calculate(Arrays.asList(new Connections[] { c }), bpvp, c.getInputLayer());

			bpAparapiOutput = bpvp.get(c.getInputLayer());
		}

		if (ffOCLOutput != null && ffAparapiOutput != null)
		{
			Tensor.TensorIterator oclIt = ffOCLOutput.iterator();
			Tensor.TensorIterator cpuIt = ffAparapiOutput.iterator();
			while (oclIt.hasNext() && cpuIt.hasNext())
			{
				assertEquals(ffAparapiOutput.getElements()[cpuIt.next()], ffOCLOutput.getElements()[oclIt.next()], 0.00001f);
			}
		}

		if (bpOCLOutput != null && bpAparapiOutput != null)
		{
			Tensor.TensorIterator oclIt = bpOCLOutput.iterator();
			Tensor.TensorIterator aparapiIt = bpAparapiOutput.iterator();
			while (oclIt.hasNext() && aparapiIt.hasNext())
			{
				assertEquals(bpAparapiOutput.getElements()[aparapiIt.next()], bpOCLOutput.getElements()[oclIt.next()], 0.00001f);
			}
		}

		/*
		 * // BP phase
		 * BackPropagationConnectionCalculator bpLRN = OperationsFactory.bpLRN(new Properties(), cc);
		 * ValuesProvider bpvp = new ValuesProvider(Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		 * bpvp.add(c.getInputLayer(), 2, 5, 2, 2);
		 * bpvp.add(c.getOutputLayer(), 2, 5, 2, 2);
		 * 
		 * bpLRN.setActivations(vp);
		 * bpLRN.calculate(Arrays.asList(new Connections[] {c}), bpvp, c.getInputLayer());
		 * 
		 * Tensor o2 = vp.get(c.getOutputLayer());
		 * assertEquals(0.791557, o2.get(0, 0, 1, 0), 0.0001);
		 * assertEquals(1.119403, o2.get(0, 1, 0, 1), 0.0001);
		 * assertEquals(1.176471, o2.get(0, 2, 0, 0), 0.0001);
		 * assertEquals(2.300406, o2.get(0, 4, 0, 0), 0.0001);
		 * assertEquals(0.791557, o2.get(1, 0, 1, 0), 0.0001);
		 * assertEquals(1.119403, o2.get(1, 1, 0, 1), 0.0001);
		 * assertEquals(1.176471, o2.get(1, 2, 0, 0), 0.0001);
		 * assertEquals(2.300406, o2.get(1, 4, 0, 0), 0.0001);
		 */
	}

	@Test
	public void testValue()
	{
		Tensor tensorInput = TensorFactory.tensor(2, 5, 2, 2);
		tensorInput.forEach(i -> tensorInput.getElements()[i] = i + 1f);
		Tensor tensorOutput = TensorFactory.tensor(2, 5, 2, 2);
		tensorOutput.forEach(i -> tensorOutput.getElements()[i] = 0f);

		// 5, 2, 2
		float[] input = tensorInput.getElements();

		int inputStartIndex = tensorInput.getStartIndex();
		int miniBatchSize = tensorInput.getDimensions()[0];
		int inputFeatureMaps = tensorInput.getDimensions()[1];
		int inputFeatureMapsLength = tensorInput.getDimensions()[2] * tensorInput.getDimensions()[3];
		int inputMiniBatchDistance = tensorInput.getDimensionElementsDistance(0);
		int inputFeatureMapsDistance = tensorInput.getDimensionElementsDistance(1);

		float[] output = tensorOutput.getElements();
		int outputStartIndex = tensorOutput.getStartIndex();

		float k = 2;
		int n = 5;
		float a = 0.01f;
		float b = 1f;

		for (int id = 0; id < 20; ++id)
		{

			// LRN computation code from aparabi.LRN.java
			int arrId = 0;
			int currentFM = id / inputFeatureMapsLength + 1;
			int startFM = (int) Math.floor(Math.max(0, currentFM - Math.floor(n / 2) - 1));
			int fmCount = (int) (Math.min(inputFeatureMaps, currentFM + Math.floor(n / 2)) - startFM);
			int startId = inputStartIndex + startFM * inputFeatureMapsDistance + (id % inputFeatureMapsLength);
			int start = 0;
			float current = 0;
			float in = 0;

			for (int i = 0; i < miniBatchSize; i++)
			{
				current = 0;
				start = startId + inputMiniBatchDistance * i;
				for (int j = 0; j < fmCount; j++)
				{
					in = input[start + j * inputFeatureMapsDistance];
					current += in * in;
				}

				arrId = id + i * inputMiniBatchDistance;

				current = k + a * current;

				output[outputStartIndex + arrId] = input[inputStartIndex + arrId] / (float) Math.pow(current, b);
			}
		}

		assertEquals(0.32573292, tensorOutput.get(0, 0, 0, 0), 0.0001);
		assertEquals(0.58823526, tensorOutput.get(0, 0, 0, 1), 0.0001);
		assertEquals(0.7915567, tensorOutput.get(0, 0, 1, 0), 0.0001);
		assertEquals(0.9433963, tensorOutput.get(0, 0, 1, 1), 0.0001);
		assertEquals(1.0504202, tensorOutput.get(0, 1, 0, 0), 0.0001);
		assertEquals(1.119403, tensorOutput.get(0, 1, 0, 1), 0.0001);
		assertEquals(1.1589404, tensorOutput.get(0, 1, 1, 0), 0.0001);
		assertEquals(1.1764706, tensorOutput.get(0, 1, 1, 1), 0.0001);
		assertEquals(1.1764705, tensorOutput.get(0, 2, 0, 0), 0.0001);
		assertEquals(1.1627907, tensorOutput.get(0, 2, 0, 1), 0.0001);
		assertEquals(1.1398964, tensorOutput.get(0, 2, 1, 0), 0.0001);
		assertEquals(1.111111, tensorOutput.get(0, 2, 1, 1), 0.0001);
		assertEquals(1.7015707, tensorOutput.get(0, 3, 0, 0), 0.0001);
		assertEquals(1.6355141, tensorOutput.get(0, 3, 0, 1), 0.0001);
		assertEquals(1.5690378, tensorOutput.get(0, 3, 1, 0), 0.0001);
		assertEquals(1.5037595, tensorOutput.get(0, 3, 1, 1), 0.0001);
		assertEquals(2.300406, tensorOutput.get(0, 4, 0, 0), 0.0001);
		assertEquals(2.195122, tensorOutput.get(0, 4, 0, 1), 0.0001);
		assertEquals(2.094818, tensorOutput.get(0, 4, 1, 0), 0.0001);
		assertEquals(2.0, tensorOutput.get(0, 4, 1, 1), 0.0001);

		assertEquals(0.99667776, tensorOutput.get(1, 0, 0, 0), 0.0001);
		assertEquals(0.9734513, tensorOutput.get(1, 0, 0, 1), 0.0001);
		assertEquals(0.9508062, tensorOutput.get(1, 0, 1, 0), 0.0001);
		assertEquals(0.92879254, tensorOutput.get(1, 0, 1, 1), 0.0001);
		assertEquals(0.7822278, tensorOutput.get(1, 1, 0, 0), 0.0001);
		assertEquals(0.76112413, tensorOutput.get(1, 1, 0, 1), 0.0001);
		assertEquals(0.740944, tensorOutput.get(1, 1, 1, 0), 0.0001);
		assertEquals(0.7216495, tensorOutput.get(1, 1, 1, 1), 0.0001);
		assertEquals(0.6352684, tensorOutput.get(1, 2, 0, 0), 0.0001);
		assertEquals(0.61728394, tensorOutput.get(1, 2, 0, 1), 0.0001);
		assertEquals(0.6001936, tensorOutput.get(1, 2, 1, 0), 0.0001);
		assertEquals(0.58394164, tensorOutput.get(1, 2, 1, 1), 0.0001);
		assertEquals(0.800194, tensorOutput.get(1, 3, 0, 0), 0.0001);
		assertEquals(0.7769653, tensorOutput.get(1, 3, 0, 1), 0.0001);
		assertEquals(0.7549612, tensorOutput.get(1, 3, 1, 0), 0.0001);
		assertEquals(0.7340947, tensorOutput.get(1, 3, 1, 1), 0.0001);
		assertEquals(1.057445, tensorOutput.get(1, 4, 0, 0), 0.0001);
		assertEquals(1.027027, tensorOutput.get(1, 4, 0, 1), 0.0001);
		assertEquals(0.99820834, tensorOutput.get(1, 4, 1, 0), 0.0001);
		assertEquals(0.9708738, tensorOutput.get(1, 4, 1, 1), 0.0001);
	}
}
