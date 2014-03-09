package com.github.neuralnetworks.samples.test;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.architecture.types.DBN;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.architecture.types.StackedAutoencoder;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorFullyConnected;
import com.github.neuralnetworks.calculation.neuronfunctions.SoftmaxFunction;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.samples.iris.IrisInputProvider;
import com.github.neuralnetworks.samples.iris.IrisTargetMultiNeuronOutputConverter;
import com.github.neuralnetworks.training.DNNLayerTrainer;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationAutoencoder;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.EarlyStoppingListener;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.rbm.AparapiCDTrainer;
import com.github.neuralnetworks.training.rbm.DBNTrainer;
import com.github.neuralnetworks.util.Environment;

/**
 * Iris test
 */
public class IrisTest {

    @Test
    public void testMLPSigmoidBP() {
	// create the network
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 4, 2, 3 }, true);

	// training and testing data providers
	IrisInputProvider trainInputProvider = new IrisInputProvider(150, 300000, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	IrisInputProvider testInputProvider = new IrisInputProvider(1, 150, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	OutputError outputError = new MultipleNeuronsOutputError();

	// trainer
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, outputError, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.5f), 0.02f, 0.7f, 0f, 0f);

	// log data
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

	// early stopping
	bpt.addEventListener(new EarlyStoppingListener(testInputProvider, 100, 0.015f));

	// execution mode
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	// train
	bpt.train();

	// add softmax function
	LayerCalculatorImpl lc = (LayerCalculatorImpl) mlp.getLayerCalculator();
	ConnectionCalculatorFullyConnected cc = (ConnectionCalculatorFullyConnected) lc.getConnectionCalculator(mlp.getOutputLayer());
	cc.addActivationFunction(new SoftmaxFunction());

	// test
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    /**
     * Contrastive Divergence testing
     */
    @Test
    public void testRBMCDSigmoidBP() {
	// RBM with 4 visible and 3 hidden units
	RBM rbm = NNFactory.rbm(4, 3, true);

	// training and testing input providers
	TrainingInputProvider trainInputProvider = new IrisInputProvider(1, 150000, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	TrainingInputProvider testInputProvider = new IrisInputProvider(1, 150, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

	// trainers
	AparapiCDTrainer t = TrainerFactory.cdSigmoidTrainer(rbm, trainInputProvider, testInputProvider, error, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.01f, 0.5f, 0f, 0f, 1, true);

	// log data
	t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

	// execution mode
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	// training
	t.train();

	// training
	t.test();

	// 2 of the iris classes are linearly not separable - an error of 1/3 illustrates that
	assertEquals(0, t.getOutputError().getTotalNetworkError(), 1/3f);
    }

    /**
     * DBN testing
     */
    @Test
    public void testDBN() {
	// deep belief network with two rbms - 4-16 and 16-3 with biases
	DBN dbn = NNFactory.dbn(new int[] {4, 4, 3}, true);
	assertEquals(2, dbn.getNeuralNetworks().size(), 0);

	dbn.setLayerCalculator(NNFactory.lcSigmoid(dbn, null));

	TrainingInputProvider trainInputProvider = new IrisInputProvider(150, 150000, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	TrainingInputProvider testInputProvider = new IrisInputProvider(1, 150, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
 
	// rbm trainers for each layer
	AparapiCDTrainer firstTrainer = TrainerFactory.cdSigmoidTrainer(dbn.getFirstNeuralNetwork(), null, null, null, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.01f, 0.5f, 0f, 0f, 1, true);
	AparapiCDTrainer lastTrainer = TrainerFactory.cdSigmoidTrainer(dbn.getLastNeuralNetwork(), null, null, null, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.01f, 0.5f, 0f, 0f, 1, true);

	Map<NeuralNetwork, OneStepTrainer<?>> map = new HashMap<>();
	map.put(dbn.getFirstNeuralNetwork(), firstTrainer);
	map.put(dbn.getLastNeuralNetwork(), lastTrainer);

	// deep trainer
	DBNTrainer deepTrainer = TrainerFactory.dbnTrainer(dbn, map, trainInputProvider, null, null);

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	// layer pre-training
	deepTrainer.train();

	// fine tuning backpropagation
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(dbn, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.01f, 0.5f, 0f, 0f);

	// log data
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

	// training
	bpt.train();

	// testing
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    @Test
    public void testAE() {
	// create autoencoder with visible layer with 4 neurons and hidden layer with 3 neurons
    	Autoencoder ae = NNFactory.autoencoderSigmoid(4, 3, true);

    	// training, testing and error
    	TrainingInputProvider trainInputProvider = new IrisInputProvider(1, 15000, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
    	TrainingInputProvider testInputProvider = new IrisInputProvider(1, 150, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
    	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

    	// backpropagation autoencoder training
    	BackPropagationAutoencoder bae = TrainerFactory.backPropagationAutoencoder(ae, trainInputProvider, testInputProvider, error, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.25f, 0.5f, 0f, 0f, 0f);

    	// log data to console
    	bae.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

    	// execution mode
    	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

    	bae.train();

    	// the output layer is needed only during the training phase...
    	ae.removeLayer(ae.getOutputLayer());

    	bae.test();

    	// 2 of the iris classes are linearly not separable - an error of 1/3 illustrates that
    	assertEquals(0, bae.getOutputError().getTotalNetworkError(), 1/3f);
    }

    @Test
    public void testSAE() {
	// create stacked autoencoder with input layer of size 4, hidden layer of the first AE with size 4 and hidden layer of the second AE with size 3
	StackedAutoencoder sae = NNFactory.saeSigmoid(new int[] { 4, 4, 3 }, true);

	// stacked networks
	Autoencoder firstNN = sae.getFirstNeuralNetwork();
	firstNN.setLayerCalculator(NNFactory.lcSigmoid(firstNN, null));

	Autoencoder lastNN = sae.getLastNeuralNetwork();
	lastNN.setLayerCalculator(NNFactory.lcSigmoid(lastNN, null));

	// trainers for each of the stacked networks
	BackPropagationAutoencoder firstTrainer = TrainerFactory.backPropagationAutoencoder(firstNN, null, null, null, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.001f, 0.5f, 0f, 0f, 0f);
	BackPropagationAutoencoder secondTrainer = TrainerFactory.backPropagationAutoencoder(lastNN, null, null, null, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.001f, 0.5f, 0f, 0f, 0f);

	Map<NeuralNetwork, OneStepTrainer<?>> map = new HashMap<>();
	map.put(firstNN, firstTrainer);
	map.put(lastNN, secondTrainer);

	// data and error providers
	TrainingInputProvider trainInputProvider = new IrisInputProvider(150, 1500000, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	TrainingInputProvider testInputProvider = new IrisInputProvider(1, 150, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

	// deep trainer
	DNNLayerTrainer deepTrainer = TrainerFactory.dnnLayerTrainer(sae, map, trainInputProvider, testInputProvider, error);

	// execution mode
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	// layerwise pre-training
	deepTrainer.train();

	// fine tuning backpropagation
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(sae, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.01f, 0.5f, 0f, 0f);

	// log data
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

	bpt.train();
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }
}
