package com.github.neuralnetworks.training.backpropagation;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.RepeaterConnection;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.backpropagation.GradientCheck;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

@RunWith(Parameterized.class)
public class GradientCheckTest extends AbstractTest
{
//	public GradientCheckTest(Runtime conf)
//	{
//		configureGlobalRuntimeEnvironment(conf);
//	}

	public GradientCheckTest(RuntimeConfiguration conf)
	{
		Environment.getInstance().setRuntimeConfiguration(conf);
	}

//	@Parameters
//	public static Collection<Runtime[]> runtimeConfigurations(){
//		List<Runtime[]> configurations = new ArrayList<>();
//		configurations.add(new Runtime[] {Runtime.CPU_SEQ});
//		configurations.add(new Runtime[] {Runtime.OPENCL});
//		configurations.add(new Runtime[] {Runtime.CPU});
//		return configurations;
//	}

	@Parameters
	public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> configurations = new ArrayList<>();

		GCRuntimeConfiguration conf1 = new GCRuntimeConfiguration();
		conf1.setRandomSeed(123456789);
		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);
		conf1.setReverseSoftmaxLoss(true);
		configurations.add(new RuntimeConfiguration[] { conf1 });

//		GCRuntimeConfiguration conf2 = new GCRuntimeConfiguration();
//		conf2.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//		conf2.setUseDataSharedMemory(true);
//		conf2.setUseWeightsSharedMemory(true);
//		conf2.setReverseSoftmaxLoss(true);
//		configurations.add(new RuntimeConfiguration[] { conf2 });

		GCRuntimeConfiguration conf3 = new GCRuntimeConfiguration();
		conf3.setRandomSeed(123456789);
		conf3.setCalculationProvider(CalculationProvider.OPENCL);
		conf3.setUseDataSharedMemory(false);
		conf3.setUseWeightsSharedMemory(false);
		conf3.setReverseSoftmaxLoss(true);
		conf3.getOpenCLConfiguration().setAggregateOperations(false);
		conf3.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
		conf3.getOpenCLConfiguration().setPushToDeviceBeforeOperation(true);
		conf3.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(false);

		conf3.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		configurations.add(new RuntimeConfiguration[] { conf3 });

		return configurations;
	}

	@Test
	public void testGradCheck0()
	{
		NeuralNetworkImpl nn = CalculationFactory.mlpSigmoid(new int[] { 2, 1 }, true);

		SimpleInputProvider inputProvider = new SimpleInputProvider(new float[][] { { 0.950129285147175f, 0.231138513574288f } }, new float[][] {{0.606842583541787f}});
		FullyConnected fc = (FullyConnected) nn.getInputLayer().getConnections().get(0);
		fc.getWeights().set(2.965287167379026f, 0, 0);
		fc.getWeights().set(-0.492512146565587f, 0, 1);

		FullyConnected b = (FullyConnected) nn.getOutputLayer().getConnections().get(1);
		b.getWeights().set(4.427042438962778f, 0);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, inputProvider, null, new MultipleNeuronsOutputError(), null, 1f, 0f, 0f, 0f, 0f, inputProvider.getInput().length,
				inputProvider.getInput().length, 1);

		GradientCheck gc = new GradientCheck(bpt, 0.001f);

		try {
			gc.compute();
		}catch (RuntimeException e){
			assertTrue(false);
		}

		assertTrue(gc.getMaxDelta() != 0);
		assertTrue(gc.getMaxDelta() != Float.NaN);
		assertTrue(gc.getMinDelta() != 0);
		assertTrue(gc.getMinDelta() != Float.NaN);
		assertTrue(gc.getMaxGradient() != 0);
		assertTrue(gc.getMaxGradient() != Float.NaN);
		assertTrue(gc.getMinGradient() != 0);
		assertTrue(gc.getMinGradient() != Float.NaN);
		assertEquals(0, gc.getMaxDelta() / inputProvider.getInput().length, 0.0001);
	}

	@Test
	public void testGradCheck1()
	{
		NeuralNetworkImpl nn = CalculationFactory.mlpSigmoid(new int[] { 2, 3, 4, 3 }, true);

		Random r = new Random();
		GCRuntimeConfiguration conf = (GCRuntimeConfiguration) Environment.getInstance().getRuntimeConfiguration();
		if (conf.getRandomSeed() != -1)
		{
			r.setSeed(conf.getRandomSeed());
		}
		new NNRandomInitializer(new RandomInitializerImpl(r, -0.5f, 0.5f)).initialize(nn);

		SimpleInputProvider inputProvider = randomInputProvider(nn, true);
		
		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, inputProvider, null, new MultipleNeuronsOutputError(), null, 1f, 0f, 0f, 0f, 0.00001f, inputProvider.getInput().length,
				inputProvider.getInput().length, 1);

		GradientCheck gc = new GradientCheck(bpt, 0.001f);

		try {
			gc.compute();
		}catch (RuntimeException e){
			assertTrue(false);
		}

		assertTrue(gc.getMaxDelta() != 0);
		assertTrue(gc.getMaxDelta() != Float.NaN);
		assertTrue(gc.getMinDelta() != 0);
		assertTrue(gc.getMinDelta() != Float.NaN);
		assertTrue(gc.getMaxGradient() != 0);
		assertTrue(gc.getMaxGradient() != Float.NaN);
		assertTrue(gc.getMinGradient() != 0);
		assertTrue(gc.getMinGradient() != Float.NaN);
		assertEquals(0, gc.getMaxDelta() / inputProvider.getInput().length, 0.0001);
	}

	@Test
	public void testGradCheck2()
	{
		NeuralNetworkImpl nn = NNFactory.mlp(new int[] { 2, 2 }, true);
		nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, OperationsFactory.softmaxFunction()));

		SimpleInputProvider inputProvider = new SimpleInputProvider(new float[][] { { 0.950129285147175f, 0.231138513574288f } }, new float[][] { { 1, 0 } });

		FullyConnected fc = (FullyConnected) nn.getInputLayer().getConnections().get(0);
		Matrix w1 = fc.getWeights();
		w1.set(-4.717681548203776f, 0, 0);
		w1.set(-0.541794167309078f, 0, 1);
		w1.set(3.149134208793002f, 1, 0);
		w1.set(1.131001410626237f, 1, 1);

		FullyConnected bc = (FullyConnected) nn.getOutputLayer().getConnections().get(1);
		Matrix wb = bc.getWeights();
		wb.set(2.568014016466234f, 0, 0);
		wb.set(-0.426528030598203f, 1, 0);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, inputProvider, null, new MultipleNeuronsOutputError(), null, 1f, 0f, 0f, 0f, 0f, inputProvider.getInput().length,
				inputProvider.getInput().length, 1);

		GradientCheck gc = new GradientCheck(bpt, 0.001f);

		try {
			gc.compute();
		}catch (RuntimeException e){
			assertTrue(false);
		}

		assertTrue(gc.getMaxDelta() != 0);
		assertTrue(gc.getMaxDelta() != Float.NaN);
		assertTrue(gc.getMinDelta() != 0);
		assertTrue(gc.getMinDelta() != Float.NaN);
		assertTrue(gc.getMaxGradient() != 0);
		assertTrue(gc.getMaxGradient() != Float.NaN);
		assertTrue(gc.getMinGradient() != 0);
		assertTrue(gc.getMinGradient() != Float.NaN);
		assertEquals(0, gc.getMaxDelta() / inputProvider.getInput().length, 0.001);
	}

	@Test
	public void testGradCheck3()
	{
		NeuralNetworkImpl nn = NNFactory.mlp(new int[] { 2, 3, 2 }, true);
		nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, OperationsFactory.softmaxFunction()));

		Random r = new Random();
		GCRuntimeConfiguration conf = (GCRuntimeConfiguration) Environment.getInstance().getRuntimeConfiguration();
		if (conf.getRandomSeed() != -1)
		{
			r.setSeed(conf.getRandomSeed());
		}
		new NNRandomInitializer(new RandomInitializerImpl(r, -0.5f, 0.5f)).initialize(nn);

		SimpleInputProvider inputProvider = randomInputProvider(nn, true);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, inputProvider, null, new MultipleNeuronsOutputError(), null, 1f, 0f, 0f, 0f, 0f, inputProvider.getInput().length,
				inputProvider.getInput().length, 1);

		GradientCheck gc = new GradientCheck(bpt, 0.001f);

		try {
			gc.compute();
		}catch (RuntimeException e){
			assertTrue(false);
		}

		assertTrue(gc.getMaxDelta() != 0);
		assertTrue(gc.getMaxDelta() != Float.NaN);
		assertTrue(gc.getMinDelta() != 0);
		assertTrue(gc.getMinDelta() != Float.NaN);
		assertTrue(gc.getMaxGradient() != 0);
		assertTrue(gc.getMaxGradient() != Float.NaN);
		assertTrue(gc.getMinGradient() != 0);
		assertTrue(gc.getMinGradient() != Float.NaN);
		assertEquals(0, gc.getMaxDelta() / inputProvider.getInput().length, 0.0001);
	}

	@Test
	public void testGradCheck4()
	{
		NeuralNetworkImpl nn = NNFactory.mlp(new int[] { 2, 4, 2 }, true);
		nn.setLayerCalculator(CalculationFactory.lcWeightedSum(nn, OperationsFactory.softmaxFunction()));

		Random r = new Random();
		GCRuntimeConfiguration conf = (GCRuntimeConfiguration) Environment.getInstance().getRuntimeConfiguration();
		if (conf.getRandomSeed() != -1)
		{
			r.setSeed(conf.getRandomSeed());
		}
		new NNRandomInitializer(new RandomInitializerImpl(r, 0f, 0.1f)).initialize(nn);

		SimpleInputProvider inputProvider = randomInputProvider(nn, true);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, inputProvider, null, new MultipleNeuronsOutputError(), null, 1f, 0f, 0f, 0f, 0f, inputProvider.getInput().length,
				inputProvider.getInput().length, 1);

		GradientCheck gc = new GradientCheck(bpt, 0.001f);

		try {
			gc.compute();
		}catch (RuntimeException e){
			assertTrue(false);
		}

		assertTrue(gc.getMaxDelta() != 0);
		assertTrue(gc.getMaxDelta() != Float.NaN);
		assertTrue(gc.getMinDelta() != 0);
		assertTrue(gc.getMinDelta() != Float.NaN);
		assertTrue(gc.getMaxGradient() != 0);
		assertTrue(gc.getMaxGradient() != Float.NaN);
		assertTrue(gc.getMinGradient() != 0);
		assertTrue(gc.getMinGradient() != Float.NaN);
		assertEquals(0, gc.getMaxDelta() / inputProvider.getInput().length, 0.0001);
	}

	@Test
	public void testGradCheck5()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 6, 6, 2 }, { 2, 2, 2, 1, 1, 0, 0 }, { 2, 2, 1, 1, 0, 0 }, { 6 }, { 2 } }, true);
		nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, null));
		CalculationFactory.lcMaxPooling(nn);

		Random r = new Random();
		GCRuntimeConfiguration conf = (GCRuntimeConfiguration) Environment.getInstance().getRuntimeConfiguration();
		if (conf.getRandomSeed() != -1)
		{
			r.setSeed(conf.getRandomSeed());
		}
		new NNRandomInitializer(new RandomInitializerImpl(r, -0.5f, 0.5f)).initialize(nn);

		SimpleInputProvider inputProvider = randomInputProvider(nn, true);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, inputProvider, null, new MultipleNeuronsOutputError(), null, 1f, 0f, 0f, 0f, 0f, inputProvider.getInput().length,
				inputProvider.getInput().length, 1);

		GradientCheck gc = new GradientCheck(bpt, 0.001f);

		try {
			gc.compute();
		}catch (RuntimeException e){
			assertTrue(false);
		}

		assertTrue(gc.getMaxDelta() != 0);
		assertTrue(gc.getMaxDelta() != Float.NaN);
		assertTrue(gc.getMinDelta() != 0);
		assertTrue(gc.getMinDelta() != Float.NaN);
		assertTrue(gc.getMaxGradient() != 0);
		assertTrue(gc.getMaxGradient() != Float.NaN);
		assertTrue(gc.getMinGradient() != 0);
		assertTrue(gc.getMinGradient() != Float.NaN);
		assertEquals(0, gc.getMaxDelta() / inputProvider.getInput().length, 0.0001);
 	}
	
	@Test
	public void testGradCheck6()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 6, 6, 2 }, { 2, 2, 2, 1, 1, 0, 0 }, { 2, 2, 1, 1, 0, 0 }, { 8 }, { 2 } }, true);
		nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, null));
		CalculationFactory.lcAveragePooling(nn);
		GCRuntimeConfiguration conf = (GCRuntimeConfiguration) Environment.getInstance().getRuntimeConfiguration();
		new NNRandomInitializer(conf.getRandomSeed() == -1 ? new RandomInitializerImpl(-0.5f, 0.5f) : new RandomInitializerImpl(-0.5f, 0.5f,conf.randomSeed)).initialize(nn);
		
		SimpleInputProvider inputProvider = randomInputProvider(nn, true);
		
		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, inputProvider, null, new MultipleNeuronsOutputError(), null, 1f, 0f, 0f, 0f, 0f, inputProvider.getInput().length,
				inputProvider.getInput().length, 1);
		
		GradientCheck gc = new GradientCheck(bpt, 0.001f);

		try {
			gc.compute();
		}catch (RuntimeException e){
			assertTrue(false);
		}
		
		assertTrue(gc.getMaxDelta() != 0);
		assertTrue(gc.getMaxDelta() != Float.NaN);
		assertTrue(gc.getMinDelta() != 0);
		assertTrue(gc.getMinDelta() != Float.NaN);
		assertTrue(gc.getMaxGradient() != 0);
		assertTrue(gc.getMaxGradient() != Float.NaN);
		assertTrue(gc.getMinGradient() != 0);
		assertTrue(gc.getMinGradient() != Float.NaN);
		assertEquals(0, gc.getMaxDelta() / inputProvider.getInput().length, 0.0001);
	}

	@Test
	public void testGradCheck7()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 8, 8, 2 }, { 2, 2, 2, 1, 1, 0, 0 }, { 2, 2, 1, 1, 0, 0 }, { 2, 2, 2, 1, 1, 0, 0 }, { 8 }, { 2 } }, true);
		nn.setLayerCalculator(CalculationFactory.lcRelu(nn, OperationsFactory.softmaxFunction()));
		CalculationFactory.lcMaxPooling(nn);
		GCRuntimeConfiguration conf = (GCRuntimeConfiguration) Environment.getInstance().getRuntimeConfiguration();
		new NNRandomInitializer(conf.getRandomSeed() == -1 ? new RandomInitializerImpl(0f, 0.09f) : new RandomInitializerImpl(0f, 0.09f, conf.getRandomSeed())).initialize(nn);
		
		SimpleInputProvider inputProvider = randomInputProvider(nn, true);
		
		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, inputProvider, null, new MultipleNeuronsOutputError(), null, 1f, 0f, 0f, 0f, 0f, inputProvider.getInput().length,
				inputProvider.getInput().length, 1);
		
		GradientCheck gc = new GradientCheck(bpt, 0.001f);

		try {
			gc.compute();
		}catch (RuntimeException e){
			assertTrue(false);
		}

		assertTrue(gc.getMaxDelta() != 0);
		assertTrue(gc.getMaxDelta() != Float.NaN);
		assertTrue(gc.getMinDelta() != 0);
		assertTrue(gc.getMinDelta() != Float.NaN);
		assertTrue(gc.getMaxGradient() != 0);
		assertTrue(gc.getMaxGradient() != Float.NaN);
		assertTrue(gc.getMinGradient() != 0);
		assertTrue(gc.getMinGradient() != Float.NaN);
		assertEquals(0, gc.getMaxDelta() / inputProvider.getInput().length, 0.0001);
	}

	@Test
	public void testGradCheckLRN()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 6, 6, 2 }, { 2, 2, 2, 1, 1, 0, 0 }, { 2, 2, 1, 1, 0, 0 }, { 8 }, { 2 } }, true);
		nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, null));
		CalculationFactory.lcMaxPooling(nn);

		Layer convLayer = nn.getInputLayer().getConnections().get(0).getOutputLayer();
		Subsampling2DConnection cs = (Subsampling2DConnection) convLayer.getConnections().get(2);

		convLayer.getConnections().remove(cs);

		Layer lrnLayer = new Layer();
		cs.setInputLayer(lrnLayer);

		new RepeaterConnection(convLayer, lrnLayer, convLayer.getUnitCount(convLayer.getConnections()));
		nn.addLayer(lrnLayer);

		LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
		lc.addConnectionCalculator(lrnLayer, OperationsFactory.lrnConnectionCalculator(2, 5, 0.0001f, 0.75f));

		new NNRandomInitializer(new RandomInitializerImpl(-0.2f, 0.2f)).initialize(nn);

		SimpleInputProvider inputProvider = randomInputProvider(nn, true);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, inputProvider, null, new MultipleNeuronsOutputError(), null, 1f, 0f, 0f, 0f, 0f, inputProvider.getInput().length,
				inputProvider.getInput().length, 1);

		GradientCheck gc = new GradientCheck(bpt, 0.001f);

		try {
			gc.compute();
		}catch (RuntimeException e){
			e.printStackTrace();
			assertTrue(false);
		}

		assertTrue(gc.getMaxDelta() != 0);
		assertTrue(gc.getMaxDelta() != Float.NaN);
		assertTrue(gc.getMinDelta() != 0);
		assertTrue(gc.getMinDelta() != Float.NaN);
		assertTrue(gc.getMaxGradient() != 0);
		assertTrue(gc.getMaxGradient() != Float.NaN);
		assertTrue(gc.getMinGradient() != 0);
		assertTrue(gc.getMinGradient() != Float.NaN);
		assertEquals(0, gc.getMaxDelta() / inputProvider.getInput().length, 0.001);
	}

	private SimpleInputProvider randomInputProvider(NeuralNetwork nn, boolean binaryOutput)
	{
		Random r = new Random();
		GCRuntimeConfiguration conf = (GCRuntimeConfiguration) Environment.getInstance().getRuntimeConfiguration();
		if (conf.getRandomSeed() != -1)
		{
			r.setSeed(conf.getRandomSeed());
		}

		int miniBatches = r.nextInt(3) + 2;
		//int miniBatches = 1;

		float[][] input = new float[miniBatches][nn.getInputLayer().getUnitCount(nn.getInputLayer().getConnections())];
		IntStream.range(0, input.length).forEach(i -> IntStream.range(0, input[i].length).forEach(j -> input[i][j] = r.nextFloat()));

		float[][] target = new float[miniBatches][nn.getOutputLayer().getUnitCount(nn.getOutputLayer().getConnections())];
		IntStream.range(0, target.length).forEach(i -> IntStream.range(0, target[i].length).forEach(j -> target[i][j] = binaryOutput ? r.nextInt(2) : r.nextFloat()));

		if (binaryOutput)
		{
			for (float[] s : target)
			{
				if (IntStream.range(0, s.length).filter(i -> s[i] != 0).count() == 0)
				{
					s[r.nextInt(s.length)] = 1;
				}

				if (IntStream.range(0, s.length).filter(i -> s[i] != 1).count() == 0)
				{
					s[r.nextInt(s.length)] = 0;
				}
			}
		}

		return new SimpleInputProvider(input, target);
	}

	private static class GCRuntimeConfiguration extends RuntimeConfiguration
	{
		private static final long serialVersionUID = 1L;

		private long randomSeed = -1;

		public long getRandomSeed()
		{
			return randomSeed;
		}

		public void setRandomSeed(long randomSeed)
		{
			this.randomSeed = randomSeed;
		}
	}
}
