package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import java.util.TreeMap;

import org.junit.Test;

import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.ConnectionFactory;
import com.github.neuralnetworks.architecture.types.DefaultNeuralNetwork;
import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSum;

/**
 * Tests for convolutional networks
 */
public class CNNTest {

    @Test
    public void testDimensions() {
	Conv2DConnection c = ConnectionFactory.convConnection(new ConvGridLayer(4, 4, 3), 2, 2, 2);

	ConvGridLayer output = (ConvGridLayer) c.getOutputLayer();

	assertEquals(3, output.getFeatureMapColumns(), 0);
	assertEquals(3, output.getFeatureMapRows(), 0);
	assertEquals(2, output.getFilters(), 0);
    }

    @Test
    public void testConvolutions() {
	DefaultNeuralNetwork nn = new DefaultNeuralNetwork();
	Conv2DConnection c = ConnectionFactory.convConnection(new ConvGridLayer(3, 3, 2), 2, 2, 2);
	nn.addConnection(c);

	Matrix i1 = new Matrix(3, 2);
	i1.set(0, 0, 1);
	i1.set(1, 0, 2);
	i1.set(2, 0, 3);
	i1.set(0, 1, 4);
	i1.set(1, 1, 5);
	i1.set(2, 1, 6);
	
	ConnectionCalculatorImpl aws = new ConnectionCalculatorImpl(new AparapiWeightedSum());

	TreeMap<Connections, Matrix> map = new TreeMap<Connections, Matrix>();
	map.put(c1, i1);
	aws.calculate(map, o, ol);

	// most simple case
	assertEquals(14, o.get(0, 0), 0);
	assertEquals(32, o.get(0, 1), 0);
	assertEquals(32, o.get(1, 0), 0);
	assertEquals(77, o.get(1, 1), 0);
	Util.fillArray(o.getElements(), 0);

	// with bias
	FullyConnected bc = new FullyConnected(new BiasLayer(), ol);
	Matrix bcg = bc.getConnectionGraph();
	bcg.set(0, 0, 0.1f);
	bcg.set(1, 0, 0.2f);
	map.put(bc, new Matrix(2, 2));
	aws = new ConnectionCalculatorImpl(new AparapiWeightedSum());
	aws.calculate(map, o, ol);

	assertEquals(14.1, o.get(0, 0), 0.01);
	assertEquals(32.1, o.get(0, 1), 0.01);
	assertEquals(32.2, o.get(1, 0), 0.01);
	assertEquals(77.2, o.get(1, 1), 0.01);
	Util.fillArray(o.getElements(), 0);

	// combined layers
	Layer il2 = new Layer(3);
	FullyConnected c2 = new FullyConnected(il2, ol);
	cg = c2.getConnectionGraph();
	cg.set(0, 0, 1);
	cg.set(0, 1, 2);
	cg.set(0, 2, 3);
	cg.set(1, 0, 4);
	cg.set(1, 1, 5);
	cg.set(1, 2, 6);

	Matrix i2 = new Matrix(3, 2);

	i2.set(0, 0, 1);
	i2.set(1, 0, 2);
	i2.set(2, 0, 3);
	i2.set(0, 1, 4);
	i2.set(1, 1, 5);
	i2.set(2, 1, 6);

	map.put(c2, i2);
	aws = new ConnectionCalculatorImpl(new AparapiWeightedSum());
	aws.calculate(map, o, ol);

	assertEquals(28.1, o.get(0, 0), 0.01);
	assertEquals(64.1, o.get(0, 1), 0.01);
	assertEquals(64.2, o.get(1, 0), 0.01);
	assertEquals(154.2, o.get(1, 1), 0.01);
	Util.fillArray(o.getElements(), 0);

    }
}
