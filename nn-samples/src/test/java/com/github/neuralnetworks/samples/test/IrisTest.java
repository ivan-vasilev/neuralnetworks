package com.github.neuralnetworks.samples.test;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.architecture.types.DBN;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.architecture.types.StackedAutoencoder;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.input.CSVInputProvider;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.MultipleNeuronsSimpleOutputError;
import com.github.neuralnetworks.input.ScalingInputFunction;
import com.github.neuralnetworks.samples.iris.IrisInputProvider;
import com.github.neuralnetworks.training.DNNLayerTrainer;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProviderImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationAutoencoder;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.training.rbm.CDTrainerBase;
import com.github.neuralnetworks.training.rbm.DBNTrainer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * Iris test
 */
@RunWith(Parameterized.class)
@Ignore
@Deprecated // Iris input is not inplemented in Core, i.e. not necessary to test.
public class IrisTest
{
	public IrisTest(RuntimeConfiguration conf)
	{
		Environment.getInstance().setRuntimeConfiguration(conf);
	}

	@Parameters
	public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> configurations = new ArrayList<>();

		RuntimeConfiguration conf1 = new RuntimeConfiguration();
		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);
		configurations.add(new RuntimeConfiguration[] { conf1 });

//		RuntimeConfiguration conf2 = new RuntimeConfiguration();
//		conf2.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//		conf2.setUseDataSharedMemory(true);
//		conf2.setUseWeightsSharedMemory(true);
//		configurations.add(new RuntimeConfiguration[] { conf2 });

//		RuntimeConfiguration conf3 = new RuntimeConfiguration();
//		conf3.setCalculationProvider(CalculationProvider.OPENCL);
//		conf3.setUseDataSharedMemory(false);
//		conf3.setUseWeightsSharedMemory(false);
//		conf3.getOpenCLConfiguration().setAggregateOperations(true);
//		conf3.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);
//		conf3.getOpenCLConfiguration().setPushToDeviceBeforeOperation(false);
//		conf3.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(true);
//		conf3.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//		configurations.add(new RuntimeConfiguration[] { conf3 });

		return configurations;
	}

	@Test
	public void testMLPSigmoidBP()
	{
		// create the network
		NeuralNetworkImpl mlp = NNFactory.mlp(new int[] { 4, 2, 3 }, true);
		mlp.setLayerCalculator(CalculationFactory.lcSigmoid(mlp, OperationsFactory.softmaxFunction()));

		// training and testing data providers
		IrisInputProvider trainInputProvider = new IrisInputProvider();
		trainInputProvider.addInputModifier(new ScalingInputFunction());
		IrisInputProvider testInputProvider = new IrisInputProvider();
		testInputProvider.addInputModifier(new ScalingInputFunction());
		OutputError outputError = new MultipleNeuronsSimpleOutputError();

		// trainer
		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, outputError, new NNRandomInitializer(new RandomInitializerImpl(-0.01f, 0.01f), 0.5f),
				0.02f, 0.7f, 0f, 0f, 0f, 150, 1, 5000);

		// log data
		bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

		// early stopping
		// bpt.addEventListener(new EarlyStoppingListener(testInputProvider, 100, 0.015f));

		// bpt.addEventListener(new FFNBPCalculationEventListener());

		// train
		bpt.train();

		// test
		bpt.test();

		assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
	}
	
	@Test
	public void testMLPReLUBP()
	{
		// create the network
		NeuralNetworkImpl mlp = NNFactory.mlp(new int[] { 4, 10, 10, 3 }, true);
		mlp.setLayerCalculator(CalculationFactory.lcRelu(mlp, OperationsFactory.softmaxFunction()));
		
		// training and testing data providers
		IrisInputProvider trainInputProvider = new IrisInputProvider();
		trainInputProvider.addInputModifier(new ScalingInputFunction());
		IrisInputProvider testInputProvider = new IrisInputProvider();
		testInputProvider.addInputModifier(new ScalingInputFunction());
		OutputError outputError = new MultipleNeuronsSimpleOutputError();

		// trainer
		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, outputError, new NNRandomInitializer(new RandomInitializerImpl(-0.01f, 0.01f), 0.5f),
				0.02f, 0.7f, 0f, 0f, 0f, 150, 1, 10000);

		// log data
		LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName());
		ls.setLogEpochs(false);
		bpt.addEventListener(ls);

		// early stopping
		// bpt.addEventListener(new EarlyStoppingListener(testInputProvider, 100, 0.015f));

		// bpt.addEventListener(new FFNBPCalculationEventListener());

		// train
		bpt.train();

		// test
		bpt.test();

		assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
	}

	@Ignore
	@Test
	public void testMLPSigmoidBPCSVReader()
	{
		// create the network
		NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 4, 2, 3 }, true);

		// training and testing data providers
		String inputPath = Thread.currentThread().getContextClassLoader().getResource("IRISinput.txt").getPath();
		String targetPath = Thread.currentThread().getContextClassLoader().getResource("IRIStarget.txt").getPath();

		TrainingInputProviderImpl trainInputProvider = new CSVInputProvider(new File(inputPath), new File(targetPath));
		trainInputProvider.addInputModifier(new ScalingInputFunction());

		TrainingInputProviderImpl testInputProvider = new CSVInputProvider(new File(inputPath), new File(targetPath));
		testInputProvider.addInputModifier(new ScalingInputFunction());

		OutputError outputError = new MultipleNeuronsOutputError();

		// trainer
		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, outputError, new NNRandomInitializer(
				new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.5f), 0.02f, 0.7f, 0f, 0f, 0f, 150, 1, 2000);

		// log data
		bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

		// early stopping
		//bpt.addEventListener(new EarlyStoppingListener(testInputProvider, 0.015f));

		// train
		bpt.train();

		// test
		bpt.test();

		assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
	}

//    @Test
//    public void testMaxout() {
//	// execution mode
//	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);
//	Environment.getInstance().setUseWeightsSharedMemory(true);
//
//	// create the network
//	NeuralNetworkImpl mlp = NNFactory.maxout(new int[] { 4, 2, 3 }, false, OperationsFactory.softmaxFunction());
//
//	// training and testing data providers
//	IrisInputProvider trainInputProvider = new IrisInputProvider(new IrisTargetMultiNeuronOutputConverter(), false);
//	trainInputProvider.addInputModifier(new ScalingInputFunction(trainInputProvider));
//	IrisInputProvider testInputProvider = new IrisInputProvider(new IrisTargetMultiNeuronOutputConverter(), false);
//	testInputProvider.addInputModifier(new ScalingInputFunction(testInputProvider));
//	OutputError outputError = new MultipleNeuronsOutputError();
//
//	// trainer
//	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, outputError, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.001f, 0.001f), 0.5f), 0.01f, 0.7f, 0f, 0f, 0f, 1, 1, 1000);
//
//	// log data
//	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));
//
//	// early stopping
//	//bpt.addEventListener(new EarlyStoppingListener(testInputProvider, 100, 0.015f));
//
//	// train
//	bpt.train();
//
//	// test
//	bpt.test();
//
//	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
//    }

	/**
	 * Contrastive Divergence testing
	 */
	@Test
	@Ignore
	public void testRBMCDSigmoidBP()
	{
		// RBM with 4 visible and 3 hidden units
		RBM rbm = NNFactory.rbm(4, 3, true);

		// training and testing input providers
		IrisInputProvider trainInputProvider = new IrisInputProvider();
		trainInputProvider.addInputModifier(new ScalingInputFunction());
		IrisInputProvider testInputProvider = new IrisInputProvider();
		testInputProvider.addInputModifier(new ScalingInputFunction());
		MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

		// trainers
		CDTrainerBase t = TrainerFactory.cdSigmoidBinaryTrainer(rbm, trainInputProvider, testInputProvider, error, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.01f,
				0.5f, 0f, 0f, 1, 1, 100, true);

		// log data
		t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

		// training
		t.train();

		// training
		t.test();

		// 2 of the iris classes are linearly not separable - an error of 2/3 illustrates that
		assertEquals(0, t.getOutputError().getTotalNetworkError(), 2 / 3f);
	}

	/**
	 * DBN testing
	 */
	@Test
	@Ignore
	public void testDBN()
	{
		// deep belief network with two rbms - 4-4 and 4-3 with biases
		DBN dbn = NNFactory.dbn(new int[] { 4, 4, 3 }, true);
		assertEquals(2, dbn.getNeuralNetworks().size(), 0);

		dbn.setLayerCalculator(CalculationFactory.lcSigmoid(dbn, null));

		IrisInputProvider trainInputProvider = new IrisInputProvider();
		trainInputProvider.addInputModifier(new ScalingInputFunction());

		IrisInputProvider testInputProvider = new IrisInputProvider();
		testInputProvider.addInputModifier(new ScalingInputFunction());

		// rbm trainers for each layer
		CDTrainerBase firstTrainer = TrainerFactory.cdSigmoidBinaryTrainer(dbn.getFirstNeuralNetwork(), null, null, null, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)),
				0.01f, 0.5f, 0f, 0f, 1, 150, 1000, true);
		CDTrainerBase lastTrainer = TrainerFactory.cdSigmoidBinaryTrainer(dbn.getLastNeuralNetwork(), null, null, null, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)),
				0.01f, 0.5f, 0f, 0f, 1, 150, 1000, true);

		Map<NeuralNetwork, OneStepTrainer<?>> map = new HashMap<>();
		map.put(dbn.getFirstNeuralNetwork(), firstTrainer);
		map.put(dbn.getLastNeuralNetwork(), lastTrainer);

		// deep trainer
		DBNTrainer deepTrainer = TrainerFactory.dbnTrainer(dbn, map, trainInputProvider, null, null);

		// layer pre-training
		deepTrainer.train();

		// fine tuning backpropagation
		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(dbn, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(
				new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.01f, 0.5f, 0f, 0f, 0f, 150, 150, 1000);

		// log data
		bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

		// training
		bpt.train();

		// testing
		bpt.test();

		assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
	}

	@Test
	@Ignore
	public void testAE()
	{
		Autoencoder ae = CalculationFactory.autoencoderSigmoid(4, 3, true);

		// training, testing and error
		IrisInputProvider trainInputProvider = new IrisInputProvider();
		trainInputProvider.addInputModifier(new ScalingInputFunction());

		IrisInputProvider testInputProvider = new IrisInputProvider();
		testInputProvider.addInputModifier(new ScalingInputFunction());

		MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

		// backpropagation autoencoder training
		BackPropagationAutoencoder bae = TrainerFactory.backPropagationAutoencoder(ae, trainInputProvider, testInputProvider, error, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f,
				0.01f)), 0.02f, 0.7f, 0f, 0f, 0f, 1, 1, 100);

		// log data to console
		bae.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

		bae.train();

		// the output layer is needed only during the training phase...
		ae.removeLayer(ae.getOutputLayer());

		bae.test();

		// 2 of the iris classes are linearly not separable - an error of 1/3 illustrates that
		assertEquals(0, bae.getOutputError().getTotalNetworkError(), 2 / 3f);
	}

	@Test
	@Ignore
	public void testSAE()
	{
		// create stacked autoencoder with input layer of size 4, hidden layer of the first AE with size 4 and hidden layer of the second AE with size 3
		StackedAutoencoder sae = CalculationFactory.saeSigmoid(new int[] { 4, 4, 3 }, true);

		// data and error providers
		IrisInputProvider trainInputProvider = new IrisInputProvider();
		trainInputProvider.addInputModifier(new ScalingInputFunction());

		IrisInputProvider testInputProvider = new IrisInputProvider();
		testInputProvider.addInputModifier(new ScalingInputFunction());

		// stacked networks
		Autoencoder firstNN = sae.getFirstNeuralNetwork();
		firstNN.setLayerCalculator(CalculationFactory.lcSigmoid(firstNN, null));

		Autoencoder lastNN = sae.getLastNeuralNetwork();
		lastNN.setLayerCalculator(CalculationFactory.lcSigmoid(lastNN, null));

		// trainers for each of the stacked networks
		BackPropagationAutoencoder firstTrainer = TrainerFactory.backPropagationAutoencoder(firstNN, null, null, null, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.5f),
				0.02f, 0.7f, 0f, 0f, 0f, 150, 1, 2000);
		BackPropagationAutoencoder secondTrainer = TrainerFactory.backPropagationAutoencoder(lastNN, null, null, null, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.5f),
				0.02f, 0.7f, 0f, 0f, 0f, 150, 1, 2000);

		Map<NeuralNetwork, OneStepTrainer<?>> map = new HashMap<>();
		map.put(firstNN, firstTrainer);
		map.put(lastNN, secondTrainer);

		// deep trainer
		DNNLayerTrainer deepTrainer = TrainerFactory.dnnLayerTrainer(sae, map, trainInputProvider, testInputProvider, null);

		// layerwise pre-training
		deepTrainer.train();

		// fine tuning backpropagation
		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(sae, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(
				new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.5f), 0.02f, 0.7f, 0f, 0f, 0f, 150, 1, 2000);

		// log data
		bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

		bpt.train();
		bpt.test();

		assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
	}
}
