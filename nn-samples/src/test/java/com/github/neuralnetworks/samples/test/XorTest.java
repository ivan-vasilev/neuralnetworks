package com.github.neuralnetworks.samples.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.PoolingLayerBuilder;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.samples.xor.XorOutputError;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.RuntimeConfiguration;

@RunWith(Parameterized.class)
@Ignore // TODO tests not working, also unit test should not be part of this project
@Deprecated
public class XorTest
{
	public XorTest(RuntimeConfiguration conf)
	{
		Environment.getInstance().setRuntimeConfiguration(conf);
	}

	@Parameters
	public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> configurations = new ArrayList<>();

//		RuntimeConfiguration conf1 = new RuntimeConfiguration();
//		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//		conf1.setUseDataSharedMemory(false);
//		conf1.setUseWeightsSharedMemory(false);
//		configurations.add(new RuntimeConfiguration[] { conf1 });

//		RuntimeConfiguration conf2 = new RuntimeConfiguration();
//		conf2.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//		conf2.setUseDataSharedMemory(true);
//		conf2.setUseWeightsSharedMemory(true);
//		configurations.add(new RuntimeConfiguration[] { conf2 });
//
//		RuntimeConfiguration conf3 = new RuntimeConfiguration();
//		conf3.setCalculationProvider(RuntimeConfiguration.CalculationProvider.OPENCL);
//		conf3.setUseDataSharedMemory(false);
//		conf3.setUseWeightsSharedMemory(false);
//		conf3.getOpenCLConfiguration().setAggregateOperations(false);
//		conf3.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
//		conf3.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//		configurations.add(new RuntimeConfiguration[] { conf3 });
//
		RuntimeConfiguration conf4 = new RuntimeConfiguration();
		conf4.setCalculationProvider(RuntimeConfiguration.CalculationProvider.OPENCL);
		conf4.setUseDataSharedMemory(false);
		conf4.setUseWeightsSharedMemory(false);
		conf4.getOpenCLConfiguration().setAggregateOperations(true);
		conf4.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);
		conf4.getOpenCLConfiguration().setUseOptionsString(true);
		conf4.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		configurations.add(new RuntimeConfiguration[] { conf4 });

		return configurations;
	}

	/**
	 * Simple xor backpropagation test
	 */
	@Test
	public void testMLPSigmoidBP()
	{
		// create training and testing input providers
		SimpleInputProvider input = new SimpleInputProvider(new float[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, new float[][] { { 0 }, { 1 }, { 1 }, { 0 } });
		SimpleInputProvider inputTest = new SimpleInputProvider(new float[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, new float[][] { { 0 }, { 1 }, { 1 }, { 0 } });

		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();
		builder.setOverrideAddBiasTo(true);
		builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 2, 1, 1));
		builder.addLayerBuilder(new FullyConnectedLayerBuilder(3).setActivationType(ActivationType.Sigmoid));
		builder.addLayerBuilder(new FullyConnectedLayerBuilder(1).setActivationType(ActivationType.Sigmoid));

		// trainer
		builder.setTrainingSet(input);
		builder.setTestingSet(inputTest);
		builder.setRand(new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)));
		builder.setError(new XorOutputError());

		builder.setLearningRate(0.1f);
		builder.setMomentum(0.9f);
		builder.setEpochs(10000);

		// create everything
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = builder.buildWithTrainer();
//		NeuralNetworkImpl mlp = neuralNetworkTrainerPair.getLeft();
		Trainer<?> bpt = neuralNetworkTrainerPair.getRight();

//  	NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 2, 2, 1 }, true);


		// create backpropagation trainer for the network
//		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, input, input, new XorOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.1f, 0.9f,
//				0f, 0f, 0f, 1, 1, 10000);

		// add logging
		bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), true, true));

		// early stopping
//		bpt.addEventListener(new EarlyStoppingListener(inputTest, 10, 0.1f));

		// train
		bpt.train();

		// test
		bpt.test();

		assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
	}
	
	/**
	 * Simple xor backpropagation test with 2 unit output layer
	 */
	@Ignore
	@Test
	public void testMLPSigmoidClassificationBP()
	{
		// create training and testing input providers
		SimpleInputProvider input = new SimpleInputProvider(new float[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, new float[][] { { 0, 1 }, { 1, 0 }, { 1, 0 }, { 0, 1 } });
		SimpleInputProvider inputTest = new SimpleInputProvider(new float[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, new float[][] { { 0, 1 }, { 1, 0 }, { 1, 0 }, { 0, 1 } });

		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();
		builder.setOverrideAddBiasTo(false);
		builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 2, 1, 1));
		builder.addLayerBuilder(new FullyConnectedLayerBuilder(4).setActivationType(ActivationType.Sigmoid));
		builder.addLayerBuilder(new FullyConnectedLayerBuilder(2).setActivationType(ActivationType.Sigmoid));

		// trainer
		builder.setTrainingSet(input);
		builder.setTestingSet(inputTest);
		builder.setRand(new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)));
		builder.setError(new MultipleNeuronsOutputError());

		builder.setLearningRate(0.1f);
		builder.setMomentum(0.9f);
		builder.setEpochs(10000);

		// create everything
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = builder.buildWithTrainer();
		Trainer<?> bpt = neuralNetworkTrainerPair.getRight();

		// add logging
		LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), true, true);
		ls.setLogBatchLoss(true);
		ls.setLogInterval(500);
		bpt.addEventListener(ls);

		// train
		bpt.train();

		// test
		bpt.test();

		assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
	}
	
	/**
	 * Simple xor backpropagation test with 2 unit softmax output layer
	 */
	@Test
	@Ignore
	public void testMLPSigmoidBPSoftmax()
	{
		// create training and testing input providers
		SimpleInputProvider input = new SimpleInputProvider(new float[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, new float[][] { { 0, 1 }, { 1, 0 }, { 1, 0 }, { 0, 1 } });
		SimpleInputProvider inputTest = new SimpleInputProvider(new float[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, new float[][] { { 0, 1 }, { 1, 0 }, { 1, 0 }, { 0, 1 } });

		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();
		builder.setOverrideAddBiasTo(true);
		builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 2, 1, 1));
		builder.addLayerBuilder(new FullyConnectedLayerBuilder(6).setActivationType(ActivationType.ReLU));
		builder.addLayerBuilder(new FullyConnectedLayerBuilder(2).setActivationType(ActivationType.SoftMax));

		// trainer
		builder.setTrainingSet(input);
		builder.setTestingSet(inputTest);
		builder.setRand(new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)));
		builder.setError(new MultipleNeuronsOutputError());

		builder.setLearningRate(0.1f);
		builder.setMomentum(0.9f);
		builder.setEpochs(1000);

		// create everything
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = builder.buildWithTrainer();
		Trainer<?> bpt = neuralNetworkTrainerPair.getRight();

		// add logging
		LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), true, true);
		ls.setLogBatchLoss(true);
		ls.setLogInterval(500);
		bpt.addEventListener(ls);

		// train
		bpt.train();

		// test
		bpt.test();

		assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
	}

	/**
	 * Simple xor feedforward test
	 */
	@Ignore
	@Test
	public void testMLPFF()
	{
		NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 2, 2, 1 }, true);

//        [-5.744886, -5.7570715, -7.329507, -7.33055] - l1-l2
//        [8.59142, 3.1430812] - bias l2
//        [12.749131, -12.848652] - l2-l3
//        [-6.1552725] - bias l3

		// weights
		FullyConnected fc1 = (FullyConnected) mlp.getInputLayer().getConnections().get(0);
		fc1.getWeights().set(-5.744886f, 0, 0);
		fc1.getWeights().set(-5.7570715f, 0, 1);
		fc1.getWeights().set(-7.329507f, 1, 0);
		fc1.getWeights().set(-7.33055f, 1, 1);

		FullyConnected b1 = (FullyConnected) fc1.getOutputLayer().getConnections().get(1);
		b1.getWeights().set(8.59142f, 0, 0);
		b1.getWeights().set(3.1430812f, 1, 0);

		FullyConnected fc2 = (FullyConnected) mlp.getOutputLayer().getConnections().get(0);
		fc2.getWeights().set(12.749131f, 0, 0);
		fc2.getWeights().set(-12.848652f, 0, 1);

		FullyConnected b2 = (FullyConnected) fc2.getOutputLayer().getConnections().get(1);
		b2.getWeights().set(-6.1552725f, 0, 0);

		// create training and testing input providers
		SimpleInputProvider input = new SimpleInputProvider(new float[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, new float[][] { { 0 }, { 1 }, { 1 }, { 0 } });

		// create backpropagation trainer for the network
		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, input, input, new XorOutputError(), null, 1f, 0.5f, 0f, 0f, 0f, 1, 1, 5000);

		// add logging
		bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

		// test
		bpt.test();

		assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
	}

	@Ignore
	@Test
	public void testCNNMLPBP()
	{
		// compare bp
		SimpleInputProvider inputProvider = new SimpleInputProvider(new float[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }, new float[][] { { 0 }, { 1 }, { 1 }, { 0 } });

		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();
		builder.setOverrideAddBiasTo(false);

		builder.addLayerBuilder(new InputLayerBuilder("input_layer", 2, 1, 1));
		builder.addLayerBuilder(new PoolingLayerBuilder(1));
		builder.addLayerBuilder(new FullyConnectedLayerBuilder(4).setActivationType(ActivationType.Sigmoid));
		builder.addLayerBuilder(new FullyConnectedLayerBuilder(1).setActivationType(ActivationType.Sigmoid));

		builder.setTrainingSet(inputProvider);
		builder.setTestingSet(inputProvider);
		builder.setError(new XorOutputError());
		builder.setRand(null);
		builder.setLearningRate(1f);
		builder.setEpochs(10000);


		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = builder.buildWithTrainer();
		NeuralNetworkImpl cnn = neuralNetworkTrainerPair.getLeft();

		FullyConnected cnnfci = (FullyConnected) cnn.getOutputLayer().getConnections().get(0).getInputLayer().getConnections().get(0);
		cnnfci.getWeights().set(0.02f, 0, 0);
		cnnfci.getWeights().set(0.01f, 1, 0);
		cnnfci.getWeights().set(0.03f, 2, 0);
		cnnfci.getWeights().set(0.001f, 3, 0);
		cnnfci.getWeights().set(0.005f, 0, 1);
		cnnfci.getWeights().set(0.04f, 1, 1);
		cnnfci.getWeights().set(0.02f, 2, 1);
		cnnfci.getWeights().set(0.009f, 3, 1);

		FullyConnected cnnfco = (FullyConnected) cnn.getOutputLayer().getConnections().get(0);
		cnnfco.getWeights().set(0.05f, 0, 0);
		cnnfco.getWeights().set(0.08f, 0, 1);

		// CNN
//		NeuralNetworkImpl cnn = NNFactory.convNN(new int[][] { { 2, 1, 1 }, { 1, 1, 1, 1, 0, 0 }, { 4 }, { 1 } }, false);
//        cnn.setLayerCalculator(CalculationFactory.lcSigmoid(cnn, null));
//        CalculationFactory.lcMaxPooling(cnn);

		// MLP
		NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 2, 4, 1 }, false);

		FullyConnected mlpfci = (FullyConnected) mlp.getOutputLayer().getConnections().get(0).getInputLayer().getConnections().get(0);
		mlpfci.getWeights().set(0.02f, 0, 0);
		mlpfci.getWeights().set(0.01f, 1, 0);
		mlpfci.getWeights().set(0.03f, 2, 0);
		mlpfci.getWeights().set(0.001f, 3, 0);
		mlpfci.getWeights().set(0.005f, 0, 1);
		mlpfci.getWeights().set(0.04f, 1, 1);
		mlpfci.getWeights().set(0.02f, 2, 1);
		mlpfci.getWeights().set(0.009f, 3, 1);

		FullyConnected mlpfco = (FullyConnected) mlp.getOutputLayer().getConnections().get(0);
		mlpfco.getWeights().set(0.05f, 0, 0);
		mlpfco.getWeights().set(0.08f, 0, 1);

		BackPropagationTrainer<?> mlpbpt = TrainerFactory.backPropagation(mlp, inputProvider, inputProvider, new XorOutputError(), null, 1f, 0f, 0f, 0f, 0f, 1, 1, 10000);
		mlpbpt.train();
		mlpbpt.test();

		BackPropagationTrainer<?> cnnbpt = TrainerFactory.backPropagation(cnn, inputProvider, inputProvider, new XorOutputError(), null, 1f, 0f, 0f, 0f, 0f, 1, 1, 10000);
		cnnbpt.train();
		cnnbpt.test();

		assertEquals(mlpbpt.getOutputError().getTotalNetworkError(), cnnbpt.getOutputError().getTotalNetworkError(), 0);
		assertTrue(Arrays.equals(cnnfco.getWeights().getElements(), mlpfco.getWeights().getElements()));
		assertTrue(Arrays.equals(cnnfci.getWeights().getElements(), mlpfci.getWeights().getElements()));
	}
}
