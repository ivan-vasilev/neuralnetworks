package com.github.neuralnetworks.samples.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.samples.xor.XorOutputError;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Environment;

public class XorTest {

    /**
     * Simple xor backpropagation test
     */
    @Test
    public void testMLPSigmoidBP() {
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	// create multi layer perceptron with one hidden layer and bias
	Environment.getInstance().setUseWeightsSharedMemory(false);
	Environment.getInstance().setUseDataSharedMemory(false);
	//NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 2, 4, 1 }, false);
	NeuralNetworkImpl mlp = NNFactory.convNN(new int[][] { { 2, 1, 1 }, { 1, 1 }, { 4 }, {1} }, false);
	//NeuralNetworkImpl mlp = NNFactory.convNN(new int[][] { {2, 1, 1}, {4}, {1} }, false);
	mlp.setLayerCalculator(NNFactory.lcSigmoid(mlp, null));
	NNFactory.lcMaxPooling(mlp);

	// create training and testing input providers
	SimpleInputProvider input = new SimpleInputProvider(new float[][] { {0, 0}, {0, 1}, {1, 0}, {1, 1} }, new float[][] { {0}, {1}, {1}, {0} });

	// create backpropagation trainer for the network
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, input, input, new XorOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 1f, 0.5f, 0f, 0f, 0f, 1, 1, 50000);

	// add logging
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

	// early stopping
	//bpt.addEventListener(new EarlyStoppingListener(testingInput, 10, 0.1f));

	// train
	bpt.train();

	// test
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    @Test
    public void testCNNMLPBP() {
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	Environment.getInstance().setUseDataSharedMemory(true);
	Environment.getInstance().setUseWeightsSharedMemory(true);

	// CNN
	NeuralNetworkImpl cnn = NNFactory.convNN(new int[][] { { 2, 1, 1 }, { 1, 1 }, { 4 }, {1} }, false);
	cnn.setLayerCalculator(NNFactory.lcSigmoid(cnn, null));
	NNFactory.lcMaxPooling(cnn);
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

	// MLP
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 2, 4, 1 }, false);

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

	// compare bp
	SimpleInputProvider inputProvider = new SimpleInputProvider(new float[][] { {0, 0}, {0, 1}, {1, 0}, {1, 1} }, new float[][] { {0}, {1}, {1}, {0} });

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
