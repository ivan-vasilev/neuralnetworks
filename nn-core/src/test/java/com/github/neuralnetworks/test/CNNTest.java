package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import java.util.TreeMap;

import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.architecture.types.ConnectionFactory;
import com.github.neuralnetworks.architecture.types.DefaultNeuralNetwork;
import com.github.neuralnetworks.calculation.ConnectionCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2D;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Util;

/**
 * Tests for convolutional networks
 */
public class CNNTest {

    @Test
    public void testDimensions() {
	// convolution dimensions
	Conv2DConnection conv = ConnectionFactory.convConnection(new ConvGridLayer(4, 4, 3), 2, 2, 2);

	ConvGridLayer output = (ConvGridLayer) conv.getOutputLayer();

	assertEquals(3, output.getFeatureMapColumns(), 0);
	assertEquals(3, output.getFeatureMapRows(), 0);
	assertEquals(2, output.getFilters(), 0);

	// subsampling dimensions
	Subsampling2DConnection sub = ConnectionFactory.subsamplingConnection(new ConvGridLayer(5, 5, 3), 2, 2);
	
	output = (ConvGridLayer) sub.getOutputLayer();

	assertEquals(2, output.getFeatureMapColumns(), 0);
	assertEquals(2, output.getFeatureMapRows(), 0);
	assertEquals(3, output.getFilters(), 0);
    }

    @Test
    public void testConvolutions() {
	DefaultNeuralNetwork nn = new DefaultNeuralNetwork();
	Conv2DConnection c = ConnectionFactory.convConnection(new ConvGridLayer(3, 3, 2), 2, 2, 1);
	nn.addConnection(c);
	c.getWeights()[0] = 1;
	c.getWeights()[1] = 2;
	c.getWeights()[2] = 3;
	c.getWeights()[3] = 4;
	c.getWeights()[4] = 1;
	c.getWeights()[5] = 2;
	c.getWeights()[6] = 3;
	c.getWeights()[7] = 4;

	Matrix i1 = new Matrix(new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}, 1);

	Matrix o = new Matrix(4, 1);

	ConnectionCalculatorImpl conv = new ConnectionCalculatorImpl(new AparapiConv2D());

	TreeMap<Connections, Matrix> map = new TreeMap<Connections, Matrix>();
	map.put(c, i1);

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);

	conv.calculate(map, o, c.getOutputLayer());

	// most simple case
	assertEquals(164, o.get(0, 0), 0);
	assertEquals(184, o.get(0, 1), 0);
	assertEquals(224, o.get(0, 2), 0);
	assertEquals(244, o.get(0, 3), 0);
	Util.fillArray(o.getElements(), 0);
    }
    
    @Test
    public void testConvolutions2() {
	DefaultNeuralNetwork nn = new DefaultNeuralNetwork();
	Conv2DConnection c = ConnectionFactory.convConnection(new ConvGridLayer(3, 3, 2), 2, 2, 2);
	nn.addConnection(c);
	c.getWeights()[0] = 1;
	c.getWeights()[1] = 2;
	c.getWeights()[2] = 3;
	c.getWeights()[3] = 4;
	c.getWeights()[4] = 1;
	c.getWeights()[5] = 2;
	c.getWeights()[6] = 3;
	c.getWeights()[7] = 4;
	c.getWeights()[8] = 1;
	c.getWeights()[9] = 2;
	c.getWeights()[10] = 3;
	c.getWeights()[11] = 4;
	c.getWeights()[12] = 1;
	c.getWeights()[13] = 2;
	c.getWeights()[14] = 3;
	c.getWeights()[15] = 4;
	
	Matrix i1 = new Matrix(new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}, 1);
	
	Matrix o = new Matrix(8, 1);
	
	ConnectionCalculatorImpl conv = new ConnectionCalculatorImpl(new AparapiConv2D());
	
	TreeMap<Connections, Matrix> map = new TreeMap<Connections, Matrix>();
	map.put(c, i1);
	
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);
	
	conv.calculate(map, o, c.getOutputLayer());
	
	assertEquals(164, o.get(0, 0), 0);
	assertEquals(184, o.get(0, 1), 0);
	assertEquals(224, o.get(0, 2), 0);
	assertEquals(244, o.get(0, 3), 0);
	assertEquals(164, o.get(0, 4), 0);
	assertEquals(184, o.get(0, 5), 0);
	assertEquals(224, o.get(0, 6), 0);
	assertEquals(244, o.get(0, 7), 0);
	Util.fillArray(o.getElements(), 0);
    }

    @Test
    public void testSubsampling	() {
    }
}
