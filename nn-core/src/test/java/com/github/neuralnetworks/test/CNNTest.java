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
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiAveragePooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiMaxPooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiStochasticPooling2D;
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
	Conv2DConnection c = ConnectionFactory.convConnection(new ConvGridLayer(3, 3, 2), 2, 2, 2);
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
    public void testPooling() {
	Subsampling2DConnection c = ConnectionFactory.subsamplingConnection(new ConvGridLayer(4, 4, 2), 2, 2);
	Matrix i1 = new Matrix(new float[] {0.5f, 1, 1, 2, 1.5f, 3, 2, 4, 2.5f, 5, 3, 6, 3.5f, 7, 4f, 8, 4.5f, 9, 5f, 10, 5.5f, 11, 6f, 12, 6.5f, 13, 7f, 14, 8f, 16, 7.5f, 15, 8.5f, 17, 9f, 18, 9.5f, 19, 10f, 20, 10.5f, 21, 11f, 22, 11.5f, 23, 12f, 24, 12.5f, 25, 13f, 26, 13.5f, 27, 14f, 28, 14.5f, 29, 15f, 30, 16f, 32, 15.5f, 31}, 2);
	TreeMap<Connections, Matrix> map = new TreeMap<Connections, Matrix>();
	map.put(c, i1);

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);

	// max pooling
	ConnectionCalculatorImpl calc = new ConnectionCalculatorImpl(new AparapiMaxPooling2D());
	Matrix o = new Matrix(8, 2);
	calc.calculate(map, o, c.getOutputLayer());

	assertEquals(3, o.get(0, 0), 0);
	assertEquals(4, o.get(1, 0), 0);
	assertEquals(7, o.get(2, 0), 0);
	assertEquals(8, o.get(3, 0), 0);
	assertEquals(11, o.get(4, 0), 0);
	assertEquals(12, o.get(5, 0), 0);
	assertEquals(15, o.get(6, 0), 0);
	assertEquals(16, o.get(7, 0), 0);

	assertEquals(6, o.get(0, 1), 0);
	assertEquals(8, o.get(1, 1), 0);
	assertEquals(14, o.get(2, 1), 0);
	assertEquals(16, o.get(3, 1), 0);
	assertEquals(22, o.get(4, 1), 0);
	assertEquals(24, o.get(5, 1), 0);
	assertEquals(30, o.get(6, 1), 0);
	assertEquals(32, o.get(7, 1), 0);

	// average pooling
	calc = new ConnectionCalculatorImpl(new AparapiAveragePooling2D());
	o = new Matrix(8, 2);
	calc.calculate(map, o, c.getOutputLayer());

	assertEquals(1.75, o.get(0, 0), 0);
	assertEquals(2.75, o.get(1, 0), 0);
	assertEquals(5.75, o.get(2, 0), 0);
	assertEquals(6.75, o.get(3, 0), 0);
	assertEquals(9.75, o.get(4, 0), 0);
	assertEquals(10.75, o.get(5, 0), 0);
	assertEquals(13.75, o.get(6, 0), 0);
	assertEquals(14.75, o.get(7, 0), 0);

	assertEquals(3.5, o.get(0, 1), 0);
	assertEquals(5.5, o.get(1, 1), 0);
	assertEquals(11.5, o.get(2, 1), 0);
	assertEquals(13.5, o.get(3, 1), 0);
	assertEquals(19.5, o.get(4, 1), 0);
	assertEquals(21.5, o.get(5, 1), 0);
	assertEquals(27.5, o.get(6, 1), 0);
	assertEquals(29.5, o.get(7, 1), 0);
    }

    @Test
    public void testStochasticPooling() {
	Subsampling2DConnection c = ConnectionFactory.subsamplingConnection(new ConvGridLayer(3, 3, 1), 3, 3);
	Matrix i1 = new Matrix(new float[] {1.6f, 0, 0, 0, 0, 0, 0, 0, 2.4f}, 2);
	TreeMap<Connections, Matrix> map = new TreeMap<Connections, Matrix>();
	map.put(c, i1);

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);

	ConnectionCalculatorImpl calc = new ConnectionCalculatorImpl(new AparapiStochasticPooling2D());
	Matrix o = new Matrix(1, 2);
	calc.calculate(map, o, c.getOutputLayer());

	assertEquals(2.08, o.get(0, 0), 0.01);
	assertEquals(2.08, o.get(0, 1), 0.01);
    }
}
