package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelsExecutor;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

@RunWith(Parameterized.class)
public class OpenCLTest
{
	public OpenCLTest(RuntimeConfiguration conf)
	{
		Environment.getInstance().setRuntimeConfiguration(conf);
	}

	@Parameters
	public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> result = new ArrayList<>();

		// The order is important!
		RuntimeConfiguration conf1 = new RuntimeConfiguration();
		conf1.setCalculationProvider(CalculationProvider.OPENCL);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);
		conf1.getOpenCLConfiguration().setAggregateOperations(true);
		conf1.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);
		result.add(new RuntimeConfiguration[] { conf1 });

		RuntimeConfiguration conf2 = new RuntimeConfiguration();
		conf2.setCalculationProvider(CalculationProvider.OPENCL);
		conf2.setUseDataSharedMemory(false);
		conf2.setUseWeightsSharedMemory(false);
		conf2.getOpenCLConfiguration().setAggregateOperations(false);
		conf2.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
		result.add(new RuntimeConfiguration[] { conf2 });

		return result;
	}
	@Ignore
	@Deprecated // test moved
	@Test
	public void testSigmoidBP()
	{
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

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 1, 0, 1 }, { 1, 1, 0 } }, new float[][] { { 1 }, { 1 } }), new SimpleInputProvider(new float[][] { { 1, 0, 1 }, { 1, 1, 0 } }, new float[][] { { 1 }, { 1 } }), new MultipleNeuronsOutputError(), null, 0.9f,
				0f, 0f, 0f, 0f, 1, 1, 1);
		bpt.train();

		if (Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getAggregateOperations()) {
			assertTrue(new String(OpenCLKernelsExecutor.getInstance().getJobs()).equals("1 !1 2 !2 3 !3 4 5 !5 6 !4 !6 7 !7 8 !8 0 !0 9 10 !10 !9 11 !11 12 !12 13 14 15 16 !13 !14 !15 !16"));
		}

		bpt.test();

		if (Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getAggregateOperations()) {
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
}
