package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Iterator;
import java.util.stream.IntStream;

import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.SoftmaxFunction;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Environment;
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
	Matrix hm1 = vp.getValues(h, 3, 2);
	Matrix hm2 = new Matrix(4, 2);
	vp.addValues(h, hm2);

	Matrix om = (Matrix) vp.getValues(o);

	assertTrue(im == vp.getValues(i, 2, 2));
	assertTrue(im == vp.getValues(i));
	assertTrue(hm1 == vp.getValues(h, 3, 2));
	assertTrue(hm2 == vp.getValues(h, 4, 2));
	assertTrue(hm1 == vp.getValues(h, nn.getConnection(i, h)));
	assertTrue(hm2 == vp.getValues(h, nn.getConnection(h, o)));
	assertTrue(om == vp.getValues(o, 1, 2));
	assertTrue(om == vp.getValues(o));
	assertTrue(2 == vp.getMiniBatchSize());
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
	Tensor t = new Tensor(5, 4, 2);
	IntStream.range(0, t.getElements().length).forEach(i -> t.getElements()[i] = i + 1);
	Matrix m = new Matrix(t, new int[][] { { 1, 1, 1 }, { 3, 2, 1 } });
	m.set(1, 0, 0);
	m.set(2, 1, 0);
	m.set(3, 2, 0);
	m.set(4, 0, 1);
	m.set(5, 1, 1);
	m.set(6, 2, 1);

	SoftmaxFunction sf = new SoftmaxFunction();
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);
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

	Iterator<Float> it = t.iterator();
	for (int i = 0; i < elements.length && it.hasNext(); i++) {
	    assertEquals(i + 1, elements[i], 0);
	    assertEquals(i + 1, it.next(), 0);
	}
	
	t = new Tensor(5, 5, 5);
	assertEquals(25, t.getDimensionElementsDistance(0), 0);
	assertEquals(5, t.getDimensionElementsDistance(1), 0);
	assertEquals(1, t.getDimensionElementsDistance(2), 0);

	elements = t.getElements();

	for (int i = 0; i < elements.length; i++) {
	    elements[i] = i + 1;
	}

	Tensor t2 = new Tensor(t, new int[][] { { 3, 0, 0 }, { 4, 4, 4 } });
	assertEquals(75, t2.getStartIndex(), 0);
	assertEquals(124, t2.getEndIndex(), 0);
	assertEquals(25, t2.getDimensionElementsDistance(0), 0);
	assertEquals(5, t2.getDimensionElementsDistance(1), 0);
	assertEquals(1, t2.getDimensionElementsDistance(2), 0);
	assertEquals(50, t2.getSize(), 0);
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
	assertEquals(6, m.getDimensionElementsDistance(0), 0);
	assertEquals(1, m.getDimensionElementsDistance(1), 0);
	assertEquals(5, m.getDimensions()[0], 0);
	assertEquals(6, m.getDimensions()[1], 0);

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
	assertEquals(2, m.get(1), 0);
	assertEquals(6, m.get(0, 5), 0);
	assertEquals(6, m.get(5), 0);

	m = new Matrix(6, 1);
	for (int i = 0; i < m.getElements().length; i++) {
	    m.getElements()[i] = i + 1;
	}

	assertEquals(2, m.get(1, 0), 0);
	assertEquals(2, m.get(1), 0);
	assertEquals(6, m.get(5, 0), 0);
	assertEquals(6, m.get(5), 0);

	// submatrix
	Tensor t = new Tensor(5, 5, 5);
	float[] elements = t.getElements();

	for (int i = 0; i < elements.length; i++) {
	    elements[i] = i + 1;
	}

	m = new Matrix(t, new int[][] { {1, 0, 0}, {1, 4, 4} });
	assertEquals(26, m.get(0, 0), 0);
	assertEquals(27, m.get(0, 1), 0);
	assertEquals(36, m.get(2, 0), 0);
	assertEquals(38, m.get(2, 2), 0);

	m = new Matrix(t, new int[][] { {1, 0, 0}, {1, 4, 4} });
	assertEquals(26, m.get(0, 0), 0);
	assertEquals(27, m.get(0, 1), 0);
	assertEquals(36, m.get(2, 0), 0);
	assertEquals(38, m.get(2, 2), 0);

	m = new Matrix(t, new int[][] { {0, 0, 1}, {4, 4, 1} });
	assertEquals(2, m.get(0, 0), 0);
	assertEquals(7, m.get(0, 1), 0);
	assertEquals(12, m.get(0, 2), 0);
	assertEquals(27, m.get(1, 0), 0);
	assertEquals(32, m.get(1, 1), 0);
	assertEquals(37, m.get(1, 2), 0);

	m = new Matrix(t, new int[][] { {2, 2, 1}, {3, 3, 1} });
	assertEquals(62, m.get(0, 0), 0);
	assertEquals(67, m.get(0, 1), 0);
	assertEquals(92, m.get(1, 1), 0);
	Iterator<Float> it = m.iterator();

	assertEquals(62, it.next(), 0);
	assertEquals(67, it.next(), 0);
	it.next();
	assertEquals(92, it.next(), 0);

	it = m.iterator(new int[][] { {1, 0}, {1, 1} });
	it.next();
	assertEquals(92, it.next(), 0);

	m = new Matrix(4, 4);
	for (int i = 0; i < m.getElements().length; i++) {
	    m.getElements()[i] = i + 1;
	}

	Matrix m2 = new Matrix(m, new int[][] { {1,1}, {2, 2} });
	assertEquals(6, m2.get(0, 0), 0);
	assertEquals(7, m2.get(0, 1), 0);
	assertEquals(10, m2.get(1, 0), 0);
	assertEquals(11, m2.get(1, 1), 0);
    }
}
