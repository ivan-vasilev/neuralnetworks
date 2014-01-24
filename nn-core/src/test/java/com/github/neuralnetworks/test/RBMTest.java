package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.rbm.PCDAparapiTrainer;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.KernelExecutionStrategy.SeqKernelExecution;

public class RBMTest {

    @Test
    public void testRBMLayerCalculator1() {
	RBM rbm = NNFactory.rbm(2, 2, false);
	NNFactory.rbmSigmoidSigmoid(rbm);

	Matrix cg1 = rbm.getMainConnections().getConnectionGraph();
	cg1.set(0, 0, 0.1f);
	cg1.set(0, 1, 0.8f);
	cg1.set(1, 0, 0.4f);
	cg1.set(1, 1, 0.6f);

	RBMLayerCalculator lc = (RBMLayerCalculator) rbm.getLayerCalculator();

	Matrix visible = new Matrix(new float[] { 0.35f, 0.9f }, 1);
	Matrix hidden = new Matrix(2, 1);
	lc.calculateHiddenLayer(rbm, visible, hidden);

	assertEquals(0.68, hidden.get(0, 0), 0.01);
	assertEquals(0.6637, hidden.get(1, 0), 0.01);
    }

    @Test
    public void testRBMLayerCalculator2() {
	RBM rbm = NNFactory.rbm(2, 2, false);
	NNFactory.rbmSigmoidSigmoid(rbm);

	Matrix cg1 = rbm.getMainConnections().getConnectionGraph();
	cg1.set(0, 0, 0.1f);
	cg1.set(1, 0, 0.8f);
	cg1.set(0, 1, 0.4f);
	cg1.set(1, 1, 0.6f);

	RBMLayerCalculator lc = (RBMLayerCalculator) rbm.getLayerCalculator();

	Matrix visible = new Matrix(2, 1);
	Matrix hidden = new Matrix(new float[] { 0.35f, 0.9f }, 1);

	lc.calculateVisibleLayer(rbm, visible, hidden);

	assertEquals(0.68, visible.get(0, 0), 0.01);
	assertEquals(0.6637, visible.get(1, 0), 0.01);
    }

    @Test
    public void testRBMLayerCalculator3() {
	RBM rbm = NNFactory.rbm(3, 2, true);
	NNFactory.rbmSigmoidSigmoid(rbm);

	Matrix cg1 = rbm.getMainConnections().getConnectionGraph();
	cg1.set(0, 0, 0.2f);
	cg1.set(0, 1, 0.4f);
	cg1.set(0, 2, -0.5f);
	cg1.set(1, 0, -0.3f);
	cg1.set(1, 1, 0.1f);
	cg1.set(1, 2, 0.2f);

	Matrix cgb1 = rbm.getHiddenBiasConnections().getConnectionGraph();
	cgb1.set(0, 0, -0.4f);
	cgb1.set(0, 1, 0.2f);

	RBMLayerCalculator lc = (RBMLayerCalculator) rbm.getLayerCalculator();
	Matrix visible = new Matrix(new float[] { 1f, 0f, 1f }, 1);
	Matrix hidden = new Matrix(2, 1);
	lc.calculateHiddenLayer(rbm, visible, hidden);

	assertEquals(0.332, hidden.get(0, 0), 0.001);
	assertEquals(0.525, hidden.get(1, 0), 0.001);
    }

    @Test
    public void testRBMLayerCalculator4() {
	RBM rbm = NNFactory.rbm(2, 3, true);
	NNFactory.rbmSigmoidSigmoid(rbm);

	Matrix cg1 = rbm.getMainConnections().getConnectionGraph();
	cg1.set(0, 0, 0.2f);
	cg1.set(1, 0, 0.4f);
	cg1.set(2, 0, -0.5f);
	cg1.set(0, 1, -0.3f);
	cg1.set(1, 1, 0.1f);
	cg1.set(2, 1, 0.2f);

	Matrix cgb1 = rbm.getVisibleBiasConnections().getConnectionGraph();
	cgb1.set(0, 0, -0.4f);
	cgb1.set(0, 1, 0.2f);

	RBMLayerCalculator lc = (RBMLayerCalculator) rbm.getLayerCalculator();
	Matrix hidden = new Matrix(new float[] { 1f, 0f, 1f }, 1);
	Matrix visible = new Matrix(2, 1);
	lc.calculateVisibleLayer(rbm, visible, hidden);

	assertEquals(0.332, visible.get(0, 0), 0.001);
	assertEquals(0.525, visible.get(1, 0), 0.001);
    }

    @Test
    public void testContrastiveDivergence() {
	RBM rbm = NNFactory.rbm(3, 2, true);

	Matrix cg1 = rbm.getMainConnections().getConnectionGraph();
	cg1.set(0, 0, 0.2f);
	cg1.set(0, 1, 0.4f);
	cg1.set(0, 2, -0.5f);
	cg1.set(1, 0, -0.3f);
	cg1.set(1, 1, 0.1f);
	cg1.set(1, 2, 0.2f);

	Matrix cgb1 = rbm.getVisibleBiasConnections().getConnectionGraph();
	cgb1.set(0, 0, 0f);
	cgb1.set(1, 0, 0f);
	cgb1.set(2, 0, 0f);

	Matrix cgb2 = rbm.getHiddenBiasConnections().getConnectionGraph();
	cgb2.set(0, 0, -0.4f);
	cgb2.set(1, 0, 0.2f);

	PCDAparapiTrainer t = TrainerFactory.pcdTrainer(rbm, new SimpleInputProvider(new float[][] { { 1, 0, 1 } }, null, 1, 1), null, null, null, 1f, 0f, 0f, 1);
	t.getProperties().setParameter(Constants.HIDDEN_CONNECTION_CALCULATOR, new AparapiSigmoid());
	t.getProperties().setParameter(Constants.VISIBLE_CONNECTION_CALCULATOR, new AparapiSigmoid());
	t.train();

	assertEquals(0.2 + 0.13203661, cg1.get(0, 0), 0.00001);
	assertEquals(0.4 - 0.22863509,  cg1.get(0, 1), 0.00001);
	assertEquals(-0.5 + 0.12887852, cg1.get(0, 2), 0.00001);
	assertEquals(-0.3 + 0.26158813, cg1.get(1, 0), 0.00001);
	assertEquals(0.1 - 0.3014404,  cg1.get(1, 1), 0.00001);
	assertEquals(0.2 + 0.25742438, cg1.get(1, 2), 0.00001);

	assertEquals(0.52276707, cgb1.get(0, 0), 0.00001);
	assertEquals(- 0.54617375, cgb1.get(1, 0), 0.00001);
	assertEquals(0.51522285, cgb1.get(2, 0), 0.00001);

	assertEquals(-0.4 - 0.08680013, cgb2.get(0, 0), 0.00001);
	assertEquals(0.2 - 0.02693379, cgb2.get(1, 0), 0.00001);
    }

    /**
     * Contrastive Divergence testing
     */
    @Test
    public void testContrastiveDivergence2() {
	RBM rbm = NNFactory.rbm(6, 2, false);
	NNFactory.rbmSigmoidSigmoid(rbm);

	TrainingInputProvider trainInputProvider = new SimpleInputProvider(new float[][] {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 0, 1} }, null, 6000, 1);
	TrainingInputProvider testInputProvider =  new SimpleInputProvider(new float[][] {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 0, 1} }, new float[][] {{1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1} }, 6, 1);
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

	PCDAparapiTrainer t = TrainerFactory.pcdTrainer(rbm, trainInputProvider, testInputProvider, error, new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.01f, 0.5f, 0f, 1);
	t.addEventListener(new LogTrainingListener());

	Environment.getInstance().setExecutionStrategy(new SeqKernelExecution());
	t.getProperties().setParameter(Constants.VISIBLE_CONNECTION_CALCULATOR, new AparapiSigmoid());
	t.getProperties().setParameter(Constants.HIDDEN_CONNECTION_CALCULATOR, new AparapiSigmoid());

	t.train();
	t.test();

	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0);
    }
}
