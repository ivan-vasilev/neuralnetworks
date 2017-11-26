package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Random;

import org.junit.Test;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelsExecutor;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Environment;

/**
 * Created by chass on 28.11.14.
 */
public class SigmoidTest extends AbstractTest
{

	@Test
	public void testSigmoidBP4()
	{
		testSigmoidBP4(Runtime.CPU_SEQ);
		testSigmoidBP4(Runtime.OPENCL);
	}

	private void testSigmoidBP4(Runtime runtime)
	{

		configureGlobalRuntimeEnvironment(runtime);

		NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 3, 2, 1 }, true);

		List<Connections> c = mlp.getConnections();
		FullyConnected c1 = (FullyConnected) c.get(0);
		Matrix cg1 = c1.getWeights();
		cg1.set(0.2f, 0, 0);
		cg1.set(0.4f, 0, 1);
		cg1.set(-0.5f, 0, 2);
		cg1.set(-0.3f, 1, 0);
		cg1.set(0.1f, 1, 1);
		cg1.set(0.2f, 1, 2);

		FullyConnected cb1 = (FullyConnected) c.get(1);
		Matrix cgb1 = cb1.getWeights();
		cgb1.set(-0.4f, 0, 0);
		cgb1.set(0.2f, 1, 0);

		FullyConnected c2 = (FullyConnected) c.get(2);
		Matrix cg2 = c2.getWeights();
		cg2.set(-0.3f, 0, 0);
		cg2.set(-0.2f, 0, 1);

		FullyConnected cb2 = (FullyConnected) c.get(3);
		Matrix cgb2 = cb2.getWeights();
		cgb2.set(0.1f, 0, 0);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 1, 0, 1 }, { 1, 1, 0 } }, new float[][] { { 1 }, { 1 } }), new SimpleInputProvider(
				new float[][] { { 1, 0, 1 }, { 1, 1, 0 } }, new float[][] { { 1 }, { 1 } }), new MultipleNeuronsOutputError(), null, 0.9f,
				0f, 0f, 0f, 0f, 1, 1, 1);
		bpt.train();

		if (Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getAggregateOperations())
		{
			assertTrue(new String(OpenCLKernelsExecutor.getInstance().getJobs()).equals("1 !1 2 !2 3 !3 4 5 !5 6 !4 !6 7 !7 8 !8 0 !0 9 10 !10 !9 11 !11 12 !12 13 14 15 16 !13 !14 !15 !16"));
		}

		bpt.test();

		if (Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getAggregateOperations())
		{
			assertTrue(new String(OpenCLKernelsExecutor.getInstance().getJobs()).equals("17 !17 18 !18 19 !19 20 21 !21 22 !20 !22 23 !23 24 !24"));
		}

		assertEquals(-0.1996, cg2.get(0, 0), 0.001);
		assertEquals(-0.0823, cg2.get(0, 1), 0.001);

		assertEquals(0.3302, cgb2.get(0, 0), 0.001);

		assertEquals(0.1849, cg1.get(0, 0), 0.001);
		assertEquals(0.3927, cg1.get(0, 1), 0.001);
		assertEquals(-0.508, cg1.get(0, 2), 0.001);
		assertEquals(-0.3098, cg1.get(1, 0), 0.001);
		assertEquals(0.0961, cg1.get(1, 1), 0.001);
		assertEquals(0.194, cg1.get(1, 2), 0.001);

		assertEquals(-0.4151, cgb1.get(0, 0), 0.001);
		assertEquals(0.1902, cgb1.get(1, 0), 0.001);
	}

	/**
	 * Simple backpropagation test with specific values
	 */
	@Test
	public void testSigmoidBP()
	{
		testSigmoidBP(Runtime.CPU_SEQ);
		testSigmoidBP(Runtime.OPENCL);
	}


	private void testSigmoidBP(Runtime runtime)
	{
		configureGlobalRuntimeEnvironment(runtime);

		NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 2, 2, 1 }, false);

		FullyConnected c1 = (FullyConnected) mlp.getInputLayer().getConnections().iterator().next();
		Matrix cg1 = c1.getWeights();
		cg1.set(0.1f, 0, 0);
		cg1.set(0.8f, 0, 1);
		cg1.set(0.4f, 1, 0);
		cg1.set(0.6f, 1, 1);

		FullyConnected c2 = (FullyConnected) mlp.getOutputLayer().getConnections().iterator().next();
		Matrix cg2 = c2.getWeights();
		cg2.set(0.3f, 0, 0);
		cg2.set(0.9f, 0, 1);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }), new SimpleInputProvider(
				new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }), new MultipleNeuronsOutputError(), null, 1f, 0f, 0f, 0f, 0f, 1, 1, 1);

		bpt.train();
		bpt.test();

		assertEquals(0.272392, cg2.get(0, 0), 0.001);
		assertEquals(0.87305, cg2.get(0, 1), 0.001);
		assertEquals(0.09916, cg1.get(0, 0), 0.01);
		assertEquals(0.7978, cg1.get(0, 1), 0.01);
		assertEquals(0.3972, cg1.get(1, 0), 0.01);
		assertEquals(0.5928, cg1.get(1, 1), 0.01);
	}

	/**
	 * Simple backpropagation test with specific values
	 * https://blog.itu.dk/MAIG-E2013/files/2013/09/9point2-classification-by-backpropagation.pdf
	 */
	@Test
	public void testSigmoidBP2()
	{
		testSigmoidBP2(Runtime.CPU_SEQ);
		testSigmoidBP2(Runtime.OPENCL);
	}

	private void testSigmoidBP2(Runtime runtime)
	{

		configureGlobalRuntimeEnvironment(runtime);

		NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 3, 2, 1 }, true);

		List<Connections> c = mlp.getConnections();
		FullyConnected c1 = (FullyConnected) c.get(0);
		Matrix cg1 = c1.getWeights();
		cg1.set(0.2f, 0, 0);
		cg1.set(0.4f, 0, 1);
		cg1.set(-0.5f, 0, 2);
		cg1.set(-0.3f, 1, 0);
		cg1.set(0.1f, 1, 1);
		cg1.set(0.2f, 1, 2);

		FullyConnected cb1 = (FullyConnected) c.get(1);
		Matrix cgb1 = cb1.getWeights();
		cgb1.set(-0.4f, 0, 0);
		cgb1.set(0.2f, 1, 0);

		FullyConnected c2 = (FullyConnected) c.get(2);
		Matrix cg2 = c2.getWeights();
		cg2.set(-0.3f, 0, 0);
		cg2.set(-0.2f, 0, 1);

		FullyConnected cb2 = (FullyConnected) c.get(3);
		Matrix cgb2 = cb2.getWeights();
		cgb2.set(0.1f, 0, 0);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 1, 0, 1 } }, new float[][] { { 1 } }), new SimpleInputProvider(new float[][] { { 1,
				0, 1 } }, new float[][] { { 1 } }), null, null, 0.9f, 0f, 0f, 0f, 0f, 1, 1, 1);
		bpt.train();

		assertEquals(-0.261, cg2.get(0, 0), 0.001);
		assertEquals(-0.138, cg2.get(0, 1), 0.001);

		assertEquals(0.218, cgb2.get(0, 0), 0.001);

		assertEquals(0.192, cg1.get(0, 0), 0.001);
		assertEquals(0.4, cg1.get(0, 1), 0.001);
		assertEquals(-0.508, cg1.get(0, 2), 0.001);
		assertEquals(-0.306, cg1.get(1, 0), 0.001);
		assertEquals(0.1, cg1.get(1, 1), 0.001);
		assertEquals(0.194, cg1.get(1, 2), 0.001);

		assertEquals(-0.408, cgb1.get(0, 0), 0.001);
		assertEquals(0.194, cgb1.get(1, 0), 0.001);
	}


	@Test
	public void testSigmoidBP3()
	{
		testSigmoidBP3(Runtime.CPU_SEQ);
		testSigmoidBP3(Runtime.OPENCL);
	}

	private void testSigmoidBP3(Runtime runtime)
	{
		configureGlobalRuntimeEnvironment(runtime);

		NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 3, 2, 1 }, true);

		List<Connections> c = mlp.getConnections();
		FullyConnected c1 = (FullyConnected) c.get(0);
		Matrix cg1 = c1.getWeights();
		cg1.set(0.2f, 0, 0);
		cg1.set(0.4f, 0, 1);
		cg1.set(-0.5f, 0, 2);
		cg1.set(-0.3f, 1, 0);
		cg1.set(0.1f, 1, 1);
		cg1.set(0.2f, 1, 2);

		FullyConnected cb1 = (FullyConnected) c.get(1);
		Matrix cgb1 = cb1.getWeights();
		cgb1.set(-0.4f, 0, 0);
		cgb1.set(0.2f, 1, 0);

		FullyConnected c2 = (FullyConnected) c.get(2);
		Matrix cg2 = c2.getWeights();
		cg2.set(-0.3f, 0, 0);
		cg2.set(-0.2f, 0, 1);

		FullyConnected cb2 = (FullyConnected) c.get(3);
		Matrix cgb2 = cb2.getWeights();
		cgb2.set(0.1f, 0, 0);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 1, 0, 1 }, { 1, 1, 0 } }, new float[][] { { 1 }, { 1 } }), null, null, null, 0.9f,
				0f, 0f, 0f, 0f, 1, 1, 1);
		bpt.train();

		assertEquals(-0.1996, cg2.get(0, 0), 0.001);
		assertEquals(-0.0823, cg2.get(0, 1), 0.001);

		assertEquals(0.3302, cgb2.get(0, 0), 0.001);

		assertEquals(0.1849, cg1.get(0, 0), 0.001);
		assertEquals(0.3927, cg1.get(0, 1), 0.001);
		assertEquals(-0.508, cg1.get(0, 2), 0.001);
		assertEquals(-0.3098, cg1.get(1, 0), 0.001);
		assertEquals(0.0961, cg1.get(1, 1), 0.001);
		assertEquals(0.194, cg1.get(1, 2), 0.001);

		assertEquals(-0.4151, cgb1.get(0, 0), 0.001);
		assertEquals(0.1902, cgb1.get(1, 0), 0.001);
	}

	@Test
	public void testSigmoid()
	{
		testSigmoid(Runtime.CPU_SEQ);
		testSigmoid(Runtime.OPENCL);
	}

	private void testSigmoid(Runtime runtime)
	{
		configureGlobalRuntimeEnvironment(runtime);

		// set to > 0 to use as constant seed
		long seed = 13265498L;

		// initialize connection weights and input
		Random r = new Random();
		if (seed > 0)
		{
			r.setSeed(seed);
		}

		Tensor m = TensorFactory.tensor(2, 3);
		m.forEach(i -> m.getElements()[i] = r.nextFloat());
		Tensor mOrig = TensorFactory.tensor(2, 3);
		mOrig.forEach(i -> mOrig.getElements()[i] = m.getElements()[i]);

		TensorFunction sigmoidFunction = OperationsFactory.sigmoidFunction();
		sigmoidFunction.value(m);

		assertTrue(OperationsFactory.isSigmoidFunction(sigmoidFunction));

		assertEquals(m.get(0, 0), 1 / (1 + Math.exp(-mOrig.get(0, 0))), 0.00001);
		assertEquals(m.get(0, 1), 1 / (1 + Math.exp(-mOrig.get(0, 1))), 0.00001);
		assertEquals(m.get(0, 2), 1 / (1 + Math.exp(-mOrig.get(0, 2))), 0.00001);
		assertEquals(m.get(1, 0), 1 / (1 + Math.exp(-mOrig.get(1, 0))), 0.00001);
		assertEquals(m.get(1, 1), 1 / (1 + Math.exp(-mOrig.get(1, 1))), 0.00001);
		assertEquals(m.get(1, 2), 1 / (1 + Math.exp(-mOrig.get(1, 2))), 0.00001);
	}

	@Test
	public void testSigmoidDerivative()
	{
		testSigmoidDerivative(Runtime.CPU_SEQ);
		testSigmoidDerivative(Runtime.OPENCL);
	}

	private void testSigmoidDerivative(Runtime runtime)
	{

		configureGlobalRuntimeEnvironment(runtime);

		// set to > 0 to use as constant seed
		long seed = 13265498L;

		// initialize connection weights and input
		Random r = new Random();
		if (seed > 0)
		{
			r.setSeed(seed);
		}

		Tensor m = TensorFactory.tensor(2, 3);
		m.forEach(i -> m.getElements()[i] = r.nextFloat());
		Tensor mOrig = TensorFactory.tensor(2, 3);
		mOrig.forEach(i -> mOrig.getElements()[i] = m.getElements()[i]);

		Tensor activation = TensorFactory.tensor(2, 3);
		new RandomInitializerImpl(r, -1.0f, 1.0f).initialize(activation);

		TensorFunction.TensorFunctionDerivative sigmoidDerivativeFunction = OperationsFactory.sigmoidDerivativeFunction();
		sigmoidDerivativeFunction.setActivations(activation);
		sigmoidDerivativeFunction.value(m);

		assertTrue(OperationsFactory.isSigmoidDerivativeFunction(sigmoidDerivativeFunction));

		assertEquals(m.get(0, 0), mOrig.get(0, 0) * activation.get(0, 0) * (1 - activation.get(0, 0)), 0.00001);
		assertEquals(m.get(0, 1), mOrig.get(0, 1) * activation.get(0, 1) * (1 - activation.get(0, 1)), 0.00001);
		assertEquals(m.get(0, 2), mOrig.get(0, 2) * activation.get(0, 2) * (1 - activation.get(0, 2)), 0.00001);
		assertEquals(m.get(1, 0), mOrig.get(1, 0) * activation.get(1, 0) * (1 - activation.get(1, 0)), 0.00001);
		assertEquals(m.get(1, 1), mOrig.get(1, 1) * activation.get(1, 1) * (1 - activation.get(1, 1)), 0.00001);
		assertEquals(m.get(1, 2), mOrig.get(1, 2) * activation.get(1, 2) * (1 - activation.get(1, 2)), 0.00001);
	}
}
