package com.github.neuralnetworks.samples.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

import com.aparapi.Kernel.EXECUTION_MODE;
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
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 2, 2, 1 }, true);

	// create training and testing input providers
	SimpleInputProvider input = new SimpleInputProvider(new float[][] { {0, 0}, {0, 1}, {1, 0}, {1, 1} }, new float[][] { {0}, {1}, {1}, {0} });

	// create backpropagation trainer for the network
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, input, input, new XorOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.1f, 0.9f, 0f, 0f, 0f, 1, 1, 100000);

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


    /**
     * Simple xor feedforward test
     */
    @Test
    public void testMLPFF() {
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	// create multi layer perceptron with one hidden layer and bias
	Environment.getInstance().setUseWeightsSharedMemory(false);
	Environment.getInstance().setUseDataSharedMemory(false);
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 2, 2, 1 }, true);

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
	SimpleInputProvider input = new SimpleInputProvider(new float[][] { {0, 0}, {0, 1}, {1, 0}, {1, 1} }, new float[][] { {0}, {1}, {1}, {0} });

	// create backpropagation trainer for the network
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, input, input, new XorOutputError(), null, 1f, 0.5f, 0f, 0f, 0f, 1, 1, 5000);

	// add logging
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

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
