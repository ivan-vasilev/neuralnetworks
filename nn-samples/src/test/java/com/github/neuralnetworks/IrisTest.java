package com.github.neuralnetworks;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Ignore;
import org.junit.Test;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.DBN;
import com.github.neuralnetworks.architecture.types.MultiLayerPerceptron;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorFullyConnected;
import com.github.neuralnetworks.calculation.neuronfunctions.SoftmaxFunction;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.samples.iris.IrisInputProvider;
import com.github.neuralnetworks.samples.iris.IrisTargetMultiNeuronOutputConverter;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProvider;
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
    @Ignore
    @Test
    public void testMLPSigmoidBP() {
	MultiLayerPerceptron mlp = NNFactory.mlpSigmoid(new int[] { 4, 2, 3 }, true);
	IrisInputProvider trainInputProvider = new IrisInputProvider(150, 1500000, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	IrisInputProvider testInputProvider = new IrisInputProvider(1, 150, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	@SuppressWarnings("unchecked")
	BackPropagationTrainer<MultiLayerPerceptron> bpt = TrainerFactory.backPropagationSigmoid(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.01f, 0.5f, 0f);

	bpt.addEventListener(new LogTrainingListener());

	Environment.getInstance().setExecutionStrategy(new SeqKernelExecution());

	bpt.train();
	LayerCalculatorImpl lc = (LayerCalculatorImpl) mlp.getLayerCalculator();
	ConnectionCalculatorFullyConnected cc = (ConnectionCalculatorFullyConnected) lc.getConnectionCalculator(mlp.getOutputLayer());
	cc.addActivationFunction(new SoftmaxFunction());

	bpt.test();
	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    /**
     * Contrastive Divergence testing
     */
    @Ignore
    @Test
    public void testRBMCDSigmoidBP() {
	RBM rbm = NNFactory.rbm(4, 3, true);
	rbm.setLayerCalculator(NNFactory.rbmSigmoidSigmoid(rbm));

	TrainingInputProvider trainInputProvider = new IrisInputProvider(1, 15000, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	TrainingInputProvider testInputProvider = new IrisInputProvider(1, 150, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

	AparapiCDTrainer t = TrainerFactory.pcdTrainer(rbm, NNFactory.rbmSigmoidSigmoid(rbm), trainInputProvider, testInputProvider, error, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.01f, 0.5f, 0f, 1);
	t.addEventListener(new LogTrainingListener());

	Environment.getInstance().setExecutionStrategy(new SeqKernelExecution());

	t.train();
	t.test();

	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0.1);
    }

    /**
     * Contrastive Divergence testing
     */
    @Test
    public void testDBN() {
	DBN dbn = NNFactory.dbn(new int[] {4, 8, 8, 3}, true);
	dbn.setLayerCalculator(NNFactory.nnSigmoid(dbn, null));

	TrainingInputProvider trainInputProvider = new IrisInputProvider(1, 15000, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	TrainingInputProvider testInputProvider = new IrisInputProvider(1, 150, new IrisTargetMultiNeuronOutputConverter(), false, true, false);
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

	AparapiCDTrainer firstTrainer = TrainerFactory.cdTrainer(dbn.getFirstNeuralNetwork(), NNFactory.rbmSigmoidSigmoid(dbn.getFirstNeuralNetwork()), null, null, null, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.01f, 0.5f, 0f, 1);
	AparapiCDTrainer secondTrainer = TrainerFactory.cdTrainer(dbn.getNeuralNetwork(1), NNFactory.rbmSigmoidSigmoid(dbn.getNeuralNetwork(1)), null, null, null, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.01f, 0.5f, 0f, 1);
	AparapiCDTrainer lastTrainer = TrainerFactory.cdTrainer(dbn.getLastNeuralNetwork(), NNFactory.rbmSigmoidSigmoid(dbn.getLastNeuralNetwork()), null, null, null, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.01f, 0.5f, 0f, 1);

	Map<NeuralNetwork, OneStepTrainer<?>> map = new HashMap<>();
	map.put(dbn.getFirstNeuralNetwork(), firstTrainer);
	map.put(dbn.getNeuralNetwork(0), secondTrainer);
	map.put(dbn.getLastNeuralNetwork(), lastTrainer);

	DBNTrainer t = TrainerFactory.dbnTrainer(dbn, map, trainInputProvider, testInputProvider, error);
	t.addEventListener(new LogTrainingListener());

	Environment.getInstance().setExecutionStrategy(new SeqKernelExecution());

	t.train();
	t.test();

	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0.1);
    }
}
