package com.github.neuralnetworks;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Ignore;
import org.junit.Test;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.architecture.types.DBN;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.architecture.types.StackedAutoencoder;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
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
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.rbm.AparapiCDTrainer;
import com.github.neuralnetworks.training.rbm.DBNTrainer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.KernelExecutionStrategy.SeqKernelExecution;

/**
 * Iris test
 */
public class IrisTest {

    /**
     * Simple iris backpropagation test
     */
    @Test
    public void testMLPSigmoidBP() {
	// create the network
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 4, 2, 3 }, true);

	// training and testing data providers
	IrisInputProvider trainInputProvider = new IrisInputProvider(150, 1500000, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	IrisInputProvider testInputProvider = new IrisInputProvider(1, 150, new IrisTargetMultiNeuronOutputConverter(), false, true, false);

	// trainer
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.01f, 0.5f, 0f);

	// log data
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

	// execution mode
	Environment.getInstance().setExecutionStrategy(new SeqKernelExecution());

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
    @Ignore
    @Test
    public void testRBMCDSigmoidBP() {
	// RBM with 4 visible and 3 hidden units
	RBM rbm = NNFactory.rbm(4, 3, true);

	TrainingInputProvider trainInputProvider = new IrisInputProvider(1, 150000, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	TrainingInputProvider testInputProvider = new IrisInputProvider(1, 150, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

	AparapiCDTrainer t = TrainerFactory.cdSigmoidTrainer(rbm, trainInputProvider, testInputProvider, error, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.01f, 0.5f, 0f, 1, true);
	t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

	Environment.getInstance().setExecutionStrategy(new SeqKernelExecution());

	t.train();
	t.test();

	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0.1);
    }

    /**
     * DBN testing
     */
    @Ignore
    @Test
    public void testDBN() {
	DBN dbn = NNFactory.dbn(new int[] {4, 8, 3}, true);

	TrainingInputProvider trainInputProvider = new IrisInputProvider(1, 150000, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	TrainingInputProvider testInputProvider = new IrisInputProvider(1, 150, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

	AparapiCDTrainer firstTrainer = TrainerFactory.cdSigmoidTrainer(dbn.getFirstNeuralNetwork(), null, null, null, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.001f, 0.5f, 0f, 1, true);
	//AparapiCDTrainer secondTrainer = TrainerFactory.cdSigmoidTrainer(dbn.getNeuralNetwork(1), null, null, null, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.01f, 0.5f, 0f, 1, true);
	AparapiCDTrainer lastTrainer = TrainerFactory.cdSigmoidTrainer(dbn.getLastNeuralNetwork(), null, null, null, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.001f, 0.5f, 0f, 1, true);

	Map<NeuralNetwork, OneStepTrainer<?>> map = new HashMap<>();
	map.put(dbn.getFirstNeuralNetwork(), firstTrainer);
	//map.put(dbn.getNeuralNetwork(1), secondTrainer);
	map.put(dbn.getLastNeuralNetwork(), lastTrainer);

	DBNTrainer t = TrainerFactory.dbnTrainer(dbn, map, trainInputProvider, testInputProvider, error);
	t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

	Environment.getInstance().setExecutionStrategy(new SeqKernelExecution());

	t.train();
	t.test();

	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0.1);
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
    	BackPropagationAutoencoder bae = TrainerFactory.backPropagationAutoencoder(ae, trainInputProvider, testInputProvider, error, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.25f, 0.5f, 0f, 0f);

    	// log data to console
    	bae.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

    	// execution mode
    	Environment.getInstance().setExecutionStrategy(new SeqKernelExecution());

    	bae.train();

    	// the output layer is needed only during the training phase...
    	ae.removeLayer(ae.getOutputLayer());

    	bae.test();

    	assertEquals(0, bae.getOutputError().getTotalNetworkError(), 0.4);
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
	BackPropagationAutoencoder firstTrainer = TrainerFactory.backPropagationAutoencoder(firstNN, null, null, null, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.001f, 0.5f, 0f, 0f);
	BackPropagationAutoencoder secondTrainer = TrainerFactory.backPropagationAutoencoder(lastNN, null, null, null, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.001f, 0.5f, 0f, 0f);

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
	Environment.getInstance().setExecutionStrategy(new SeqKernelExecution());

	// layerwise pre-training
	deepTrainer.train();

	// fine tuning backpropagation
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(sae, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.01f, 0.5f, 0f);

	// log data
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

	bpt.train();
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }
}
