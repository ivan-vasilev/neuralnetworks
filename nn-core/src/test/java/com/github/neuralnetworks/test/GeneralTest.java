package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.SoftmaxFunction;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Util;

public class GeneralTest {

    @Test
    public void testValuesProvider() {
	ValuesProvider vp = new ValuesProvider();

	NeuralNetworkImpl nn = new NeuralNetworkImpl();;
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
    }

    @Test
    public void testRandomInitializer() {
	NeuralNetworkImpl nn = NNFactory.mlp(new int[] {3, 2}, true);
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
}
