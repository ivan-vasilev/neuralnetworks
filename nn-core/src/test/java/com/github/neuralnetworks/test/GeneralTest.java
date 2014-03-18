package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.SoftmaxFunction;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Matrix;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.Util;

public class GeneralTest {

    @Test
    public void testValuesProvider() {
	ValuesProvider vp = new ValuesProvider();

	NeuralNetworkImpl nn = new NeuralNetworkImpl();
	Layer i = new Layer();
	Layer h = new Layer();
	Layer o = new Layer();

	nn.addLayer(i);

	NNFactory.addFullyConnectedLayer(nn, h, 2, 3, true);
	NNFactory.addFullyConnectedLayer(nn, o, 4, 1, true);

	Matrix im = new Matrix(2, 2);
	vp.addValues(i, im);
	Matrix hm1 = vp.getValues(h, 3);
	Matrix hm2 = new Matrix(4, 2);
	vp.addValues(h, hm2);

	Matrix om = vp.getValues(o);

	assertTrue(im == vp.getValues(i, 2));
	assertTrue(im == vp.getValues(i));
	assertTrue(hm1 == vp.getValues(h, 3));
	assertTrue(hm2 == vp.getValues(h, 4));
	assertTrue(hm1 == vp.getValues(h, nn.getConnection(i, h)));
	assertTrue(hm2 == vp.getValues(h, nn.getConnection(h, o)));
	assertTrue(om == vp.getValues(o, 1));
	assertTrue(om == vp.getValues(o));
	assertTrue(2 == vp.getColumns());
    }

    @Test
    public void testRandomInitializer() {
	NeuralNetworkImpl nn = NNFactory.mlp(new int[] { 3, 2 }, true);
	NNRandomInitializer rand = new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.1f, 0.1f), 0.5f);
	rand.initialize(nn);

	for (Layer l : nn.getLayers()) {
	    if (Util.isBias(l)) {
		GraphConnections gc = (GraphConnections) l.getConnections().get(0);
		for (float v : gc.getConnectionGraph().getElements()) {
		    assertEquals(0.5, v, 0f);
		}
	    } else {
		GraphConnections gc = (GraphConnections) l.getConnections().get(0);
		for (float v : gc.getConnectionGraph().getElements()) {
		    assertTrue(v >= -0.1f && v <= 0.1f && v != 0);
		}
	    }
	}

	rand = new NNRandomInitializer(new MersenneTwisterRandomInitializer(2f, 3f), new MersenneTwisterRandomInitializer(-2f, -1f));
	rand.initialize(nn);

	for (Layer l : nn.getLayers()) {
	    if (Util.isBias(l)) {
		GraphConnections gc = (GraphConnections) l.getConnections().get(0);
		for (float v : gc.getConnectionGraph().getElements()) {
		    assertTrue(v >= -2f && v <= -1f);
		}
	    } else {
		GraphConnections gc = (GraphConnections) l.getConnections().get(0);
		for (float v : gc.getConnectionGraph().getElements()) {
		    assertTrue(v >= 2f && v <= 3f);
		}
	    }
	}
    }

    @Test
    public void testSoftMax() {
	SoftmaxFunction sf = new SoftmaxFunction();
	Matrix m = new Matrix(3, 2);
	m.set(0, 0, 1);
	m.set(1, 0, 2);
	m.set(2, 0, 3);
	m.set(0, 1, 4);
	m.set(1, 1, 5);
	m.set(2, 1, 6);

	sf.value(m);

	assertEquals(1 / 6f, m.get(0, 0), 0);
	assertEquals(2 / 6f, m.get(1, 0), 0);
	assertEquals(3 / 6f, m.get(2, 0), 0);
	assertEquals(4 / 15f, m.get(0, 1), 0);
	assertEquals(5 / 15f, m.get(1, 1), 0);
	assertEquals(6 / 15f, m.get(2, 1), 0);
    }

    @Test
    public void testTensor() {
	Tensor t = new Tensor(2, 2, 2);
	float[] elements = t.getElements();

	assertEquals(8, elements.length, 0);

	t.set(1, 0, 0, 0);
	t.set(2, 0, 0, 1);
	t.set(3, 0, 1, 0);
	t.set(4, 0, 1, 1);
	t.set(5, 1, 0, 0);
	t.set(6, 1, 0, 1);
	t.set(7, 1, 1, 0);
	t.set(8, 1, 1, 1);

	for (int i = 0; i < elements.length; i++) {
	    assertEquals(i + 1, elements[i], 0);
	}

	t = new Tensor(5, 5, 5);
	elements = t.getElements();

	for (int i = 0; i < elements.length; i++) {
	    elements[i] = i + 1;
	}

	Tensor t2 = new Tensor(t, new int[] { 3, 0, 0 }, new int[] { 4, 4, 4 });

	assertEquals(76, t2.get(0, 0, 0), 0);
	assertEquals(77, t2.get(0, 0, 1), 0);
	assertEquals(81, t2.get(0, 1, 0), 0);
	assertEquals(101, t2.get(1, 0, 0), 0);
	assertEquals(106, t2.get(1, 1, 0), 0);
	assertEquals(112, t2.get(1, 2, 1), 0);
    }

    @Test
    public void testMatrix() {
	Matrix m = new Matrix(5, 6);

	assertEquals(5, m.getRows(), 0);
	assertEquals(6, m.getColumns(), 0);

	for (int i = 0; i < m.getElements().length; i++) {
	    m.getElements()[i] = i + 1;
	}

	assertEquals(2, m.get(0, 1), 0);
	assertEquals(15, m.get(2, 2), 0);

	m = new Matrix(1, 6);
	for (int i = 0; i < m.getElements().length; i++) {
	    m.getElements()[i] = i + 1;
	}

	assertEquals(2, m.get(0, 1), 0);
	assertEquals(6, m.get(0, 5), 0);


	m = new Matrix(6, 1);
	for (int i = 0; i < m.getElements().length; i++) {
	    m.getElements()[i] = i + 1;
	}

	assertEquals(2, m.get(1, 0), 0);
	assertEquals(6, m.get(5, 0), 0);
    }
}
