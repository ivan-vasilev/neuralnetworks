package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.rbm.AparapiCDTrainer;
import com.github.neuralnetworks.util.Environment;

public class RBMTest {

    /**
     * Contrastive Divergence testing
     */
    @Test
    public void testContrastiveDivergence() {
	// RBM with 6 visible, 2 hidden units and a bias
	RBM rbm = NNFactory.rbm(6, 2, true);

	// We'll use a simple dataset of symptoms of a flu illness. There are 6
	// input features and the first three are symptoms of the illness - for
	// example 1 0 0 0 0 0 means that a patient has high temperature, 0 1
	// 0 0 0 0 - coughing, 1 1 0 0 0 0 - coughing and high temperature
	// and so on. The second three features are "counter" symptomps - when a
	// patient has one of those it is less likely that he's sick. For
	// example 0 0 0 1 0 0 means that he has a flu vaccine. It's possible
	// to have combinations between both - for exmample 0 1 0 1 0 0 means
	// that the patient is vaccinated, but he's also coughing. We will
	// consider a patient to be sick when he has at least two of the first
	// three and healthy if he has two of the second three
	TrainingInputProvider trainInputProvider = new SimpleInputProvider(new float[][] { { 1, 1, 1, 0, 0, 0 }, { 1, 0, 1, 0, 0, 0 }, { 1, 1, 0, 0, 0, 0 }, { 0, 1, 1, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0 }, { 0, 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1, 0 }, { 0, 0, 0, 1, 0, 1 }, { 0, 0, 0, 0, 1, 1 }, { 0, 0, 0, 1, 1, 0 } }, null, 1000, 1);
	TrainingInputProvider testInputProvider = new SimpleInputProvider(new float[][] { { 1, 1, 1, 0, 0, 0 }, { 1, 0, 1, 0, 0, 0 }, { 1, 1, 0, 0, 0, 0 }, { 0, 1, 1, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0 }, { 0, 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1, 0 }, { 0, 0, 0, 1, 0, 1 }, { 0, 0, 0, 0, 1, 1 }, { 0, 0, 0, 1, 1, 0 } }, new float[][] { { 1, 0 }, { 1, 0 }, { 1, 0 }, { 1, 0 }, { 1, 0 }, { 0, 1 }, { 0, 1 }, { 0, 1 }, { 0, 1 }, { 0, 1 } }, 10, 1);
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

	// Contrastive divergence training
	AparapiCDTrainer t = TrainerFactory.cdSigmoidTrainer(rbm, trainInputProvider, testInputProvider, error, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.02f, 0.5f, 0f, 0f, 1, false);
	t.setLayerCalculator(NNFactory.rbmSigmoidSigmoid(rbm));

	// log data
	t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), true, false));

	// training
	t.train();

	// testing
	t.test();

	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0);
    }
    
    /**
     * Persistent Contrastive Divergence testing
     */
    @Test
    public void testPersistentContrastiveDivergence() {
	// RBM with 6 visible, 2 hidden units and bias
	RBM rbm = NNFactory.rbm(6, 2, true);
	
	// We'll use a simple dataset of symptoms of a flu illness. There are 6
	// input features and the first three are symptoms of the illness - for
	// example 1 0 0 0 0 0 means that a patient has high temperature, 0 1
	// 0 0 0 0 - coughing, 1 1 0 0 0 0 - coughing and high temperature
	// and so on. The second three features are "counter" symptomps - when a
	// patient has one of those it is less likely that he's sick. For
	// example 0 0 0 1 0 0 means that he has a flu vaccine. It's possible
	// to have combinations between both - for exmample 0 1 0 1 0 0 means
	// that the patient is vaccinated, but he's also coughing. We will
	// consider a patient to be sick when he has at least two of the first
	// three and healthy if he has two of the second three
	TrainingInputProvider trainInputProvider = new SimpleInputProvider(new float[][] { { 1, 1, 1, 0, 0, 0 }, { 1, 0, 1, 0, 0, 0 }, { 1, 1, 0, 0, 0, 0 }, { 0, 1, 1, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0 }, { 0, 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1, 0 }, { 0, 0, 0, 1, 0, 1 }, { 0, 0, 0, 0, 1, 1 }, { 0, 0, 0, 1, 1, 0 } }, null, 1000, 1);
	TrainingInputProvider testInputProvider = new SimpleInputProvider(new float[][] { { 1, 1, 1, 0, 0, 0 }, { 1, 0, 1, 0, 0, 0 }, { 1, 1, 0, 0, 0, 0 }, { 0, 1, 1, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0 }, { 0, 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1, 0 }, { 0, 0, 0, 1, 0, 1 }, { 0, 0, 0, 0, 1, 1 }, { 0, 0, 0, 1, 1, 0 } }, new float[][] { { 1, 0 }, { 1, 0 }, { 1, 0 }, { 1, 0 }, { 1, 0 }, { 0, 1 }, { 0, 1 }, { 0, 1 }, { 0, 1 }, { 0, 1 } }, 10, 1);
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

	// Persistent Contrastive divergence trainer
	AparapiCDTrainer t = TrainerFactory.cdSigmoidTrainer(rbm, trainInputProvider, testInputProvider, error, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.02f, 0.5f, 0f, 0f, 1, true);
	t.setLayerCalculator(NNFactory.rbmSigmoidSigmoid(rbm));

	// log data
	t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), true, false));

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	// training
	t.train();

	// testing
	t.test();
	
	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0);
    }

    @Test
    public void testRBMLayerCalculator1() {
	RBM rbm = NNFactory.rbm(2, 2, false);
	rbm.setLayerCalculator(NNFactory.rbmSigmoidSigmoid(rbm));

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
	rbm.setLayerCalculator(NNFactory.rbmSigmoidSigmoid(rbm));

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
	rbm.setLayerCalculator(NNFactory.rbmSigmoidSigmoid(rbm));

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
	rbm.setLayerCalculator(NNFactory.rbmSigmoidSigmoid(rbm));

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
    public void testOneStepContrastiveDivergence() {
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

	AparapiCDTrainer t = TrainerFactory.cdSigmoidTrainer(rbm, new SimpleInputProvider(new float[][] { { 1, 0, 1 } }, null, 1, 1), null, null, null, 1f, 0f, 0f, 0f, 1, true);
	t.setLayerCalculator(NNFactory.rbmSigmoidSigmoid(rbm));

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	t.train();

	assertEquals(0.52276707, cgb1.get(0, 0), 0.00001);
	assertEquals(- 0.54617375, cgb1.get(1, 0), 0.00001);
	assertEquals(0.51522285, cgb1.get(2, 0), 0.00001);
	
	assertEquals(-0.4 - 0.08680013, cgb2.get(0, 0), 0.00001);
	assertEquals(0.2 - 0.02693379, cgb2.get(1, 0), 0.00001);

	assertEquals(0.2 + 0.13203661, cg1.get(0, 0), 0.00001);
	assertEquals(0.4 - 0.22863509,  cg1.get(0, 1), 0.00001);
	assertEquals(-0.5 + 0.12887852, cg1.get(0, 2), 0.00001);
	assertEquals(-0.3 + 0.26158813, cg1.get(1, 0), 0.00001);
	assertEquals(0.1 - 0.3014404,  cg1.get(1, 1), 0.00001);
	assertEquals(0.2 + 0.25742438, cg1.get(1, 2), 0.00001);
    }
}
