package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;

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
}
