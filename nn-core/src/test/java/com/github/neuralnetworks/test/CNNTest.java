package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import org.junit.Test;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiAveragePooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DFF;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiMaxPooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiStochasticPooling2D;
import com.github.neuralnetworks.training.backpropagation.BackpropagationAveragePooling2D;
import com.github.neuralnetworks.training.backpropagation.BackpropagationMaxPooling2D;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.KernelExecutionStrategy.SeqKernelExecution;
import com.github.neuralnetworks.util.Util;

/**
 * Tests for convolutional networks
 */
public class CNNTest {

    @Test
    public void testDimensions() {
	// convolution dimensions
	Conv2DConnection conv = new Conv2DConnection(new ConvGridLayer(4, 4, 3), 2, 2, 2);

	ConvGridLayer output = (ConvGridLayer) conv.getOutputLayer();

	assertEquals(3, output.getFeatureMapColumns(), 0);
	assertEquals(3, output.getFeatureMapRows(), 0);
	assertEquals(2, output.getFilters(), 0);

	// subsampling dimensions
	Subsampling2DConnection sub = new Subsampling2DConnection(new ConvGridLayer(5, 5, 3), 2, 2);

	output = (ConvGridLayer) sub.getOutputLayer();

	assertEquals(2, output.getFeatureMapColumns(), 0);
	assertEquals(2, output.getFeatureMapRows(), 0);
	assertEquals(3, output.getFilters(), 0);
    }

    @Test
    public void testConvolutions() {
	NeuralNetworkImpl nn = new NeuralNetworkImpl();
	Conv2DConnection c = new Conv2DConnection(new ConvGridLayer(3, 3, 2), 2, 2, 1);
	nn.addConnection(c);
	c.getWeights()[0] = 1;
	c.getWeights()[1] = 2;
	c.getWeights()[2] = 3;
	c.getWeights()[3] = 4;
	c.getWeights()[4] = 1;
	c.getWeights()[5] = 2;
	c.getWeights()[6] = 3;
	c.getWeights()[7] = 4;

	Matrix i1 = new Matrix(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 }, 1);

	Matrix o = new Matrix(4, 1);

	AparapiConv2D conv = new AparapiConv2DFF(c, 1);

	conv.calculate(c, i1, o);

	// most simple case
	assertEquals(164, o.get(0, 0), 0);
	assertEquals(184, o.get(0, 1), 0);
	assertEquals(224, o.get(0, 2), 0);
	assertEquals(244, o.get(0, 3), 0);
	Util.fillArray(o.getElements(), 0);
    }

    @Test
    public void testConvolutions2() {
	Conv2DConnection c = new Conv2DConnection(new ConvGridLayer(3, 3, 2), 2, 2, 2);
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

	Matrix i1 = new Matrix(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 }, 1);

	Matrix o = new Matrix(8, 1);

	AparapiConv2D conv = new AparapiConv2DFF(c, 1);

	conv.calculate(c, i1, o);

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
    public void testMaxPooling() {
	Subsampling2DConnection c = new Subsampling2DConnection(new ConvGridLayer(4, 4, 2), 2, 2);
	Matrix i1 = new Matrix(new float[] { 0.5f, 1, 1, 2, 1.5f, 3, 2, 4, 2.5f, 5, 3, 6, 3.5f, 7, 4f, 8, 4.5f, 9, 5f, 10, 5.5f, 11, 6f, 12, 6.5f, 13, 7f, 14, 8f, 16, 7.5f, 15, 8.5f, 17, 9f, 18, 9.5f, 19, 10f, 20, 10.5f, 21, 11f, 22, 11.5f, 23, 12f, 24, 12.5f, 25, 13f, 26, 13.5f, 27, 14f, 28, 14.5f, 29, 15f, 30, 16f, 32, 15.5f, 31 }, 2);
	TreeMap<Connections, Matrix> map = new TreeMap<Connections, Matrix>();
	map.put(c, i1);

	ConnectionCalculator calc = new AparapiMaxPooling2D(c, 2);
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
    }

    @Test
    public void testAveragePooling() {
	Subsampling2DConnection c = new Subsampling2DConnection(new ConvGridLayer(4, 4, 2), 2, 2);
	Matrix i1 = new Matrix(new float[] { 0.5f, 1, 1, 2, 1.5f, 3, 2, 4, 2.5f, 5, 3, 6, 3.5f, 7, 4f, 8, 4.5f, 9, 5f, 10, 5.5f, 11, 6f, 12, 6.5f, 13, 7f, 14, 8f, 16, 7.5f, 15, 8.5f, 17, 9f, 18, 9.5f, 19, 10f, 20, 10.5f, 21, 11f, 22, 11.5f, 23, 12f, 24, 12.5f, 25, 13f, 26, 13.5f, 27, 14f, 28, 14.5f, 29, 15f, 30, 16f, 32, 15.5f, 31 }, 2);
	TreeMap<Connections, Matrix> map = new TreeMap<Connections, Matrix>();
	map.put(c, i1);

	AparapiAveragePooling2D calc = new AparapiAveragePooling2D(c, 2);
	Matrix o = new Matrix(8, 2);
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
	Subsampling2DConnection c = new Subsampling2DConnection(new ConvGridLayer(3, 3, 1), 3, 3);
	Matrix i1 = new Matrix(new float[] { 1.6f, 1.6f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.4f, 2.4f }, 2);
	TreeMap<Connections, Matrix> map = new TreeMap<Connections, Matrix>();
	map.put(c, i1);

	AparapiStochasticPooling2D calc = new AparapiStochasticPooling2D(c, 2);
	Matrix o = new Matrix(1, 2);
	calc.calculate(map, o, c.getOutputLayer());

	assertEquals(2.08, o.get(0, 0), 0.01);
	assertEquals(2.08, o.get(0, 1), 0.01);
    }

    @Test
    public void testMaxPoolingBackpropagation() {
	Subsampling2DConnection c = new Subsampling2DConnection(new ConvGridLayer(4, 4, 2), 2, 2);
	Matrix a1 = new Matrix(new float[] { 0.5f, 1, 1, 2, 1.5f, 3, 2, 4, 2.5f, 5, 3, 6, 3.5f, 7, 4f, 8, 4.5f, 9, 5f, 10, 5.5f, 11, 6f, 12, 6.5f, 13, 7f, 14, 8f, 16, 7.5f, 15, 8.5f, 17, 9f, 18, 9.5f, 19, 10f, 20, 10.5f, 21, 11f, 22, 11.5f, 23, 12f, 24, 12.5f, 25, 13f, 26, 13.5f, 27, 14f, 28, 14.5f, 29, 15f, 30, 16f, 32, 15.5f, 31 }, 2);

	// max pooling
	ConnectionCalculator calc = new AparapiMaxPooling2D(c, 2);
	TreeMap<Connections, Matrix> map = new TreeMap<Connections, Matrix>();
	map.put(c, a1);
	Matrix o = new Matrix(8, 2);
	calc.calculate(map, o, c.getOutputLayer());

	Map<Layer, Matrix> activations = new HashMap<Layer, Matrix>();
	activations.put(c.getInputLayer(), a1);

	BackpropagationMaxPooling2D bp = new BackpropagationMaxPooling2D(c, 2);
	bp.setActivations(activations);

	Matrix bpo = new Matrix(32, 2);
	map.put(c, o);
	bp.calculate(map, bpo, c.getInputLayer());

	assertEquals(true, bpo.get(5, 0) == a1.get(5, 0));
	assertEquals(true, bpo.get(7, 0) == a1.get(7, 0));
	assertEquals(true, bpo.get(13, 0) == a1.get(13, 0));
	assertEquals(true, bpo.get(14, 0) == a1.get(14, 0));
	assertEquals(true, bpo.get(5, 1) == a1.get(5, 1));
	assertEquals(true, bpo.get(7, 1) == a1.get(7, 1));
	assertEquals(true, bpo.get(13, 1) == a1.get(13, 1));
	assertEquals(true, bpo.get(14, 1) == a1.get(14, 1));
    }

    @Test
    public void testAveragePoolingBackpropagation() {
	Environment.getInstance().setExecutionStrategy(new SeqKernelExecution());
	Subsampling2DConnection c = new Subsampling2DConnection(new ConvGridLayer(4, 4, 2), 2, 2);
	Matrix a1 = new Matrix(new float[] { 0.5f, 1, 1, 2, 1.5f, 3, 2, 4, 2.5f, 5, 3, 6, 3.5f, 7, 4f, 8, 4.5f, 9, 5f, 10, 5.5f, 11, 6f, 12, 6.5f, 13, 7f, 14, 8f, 16, 7.5f, 15, 8.5f, 17, 9f, 18, 9.5f, 19, 10f, 20, 10.5f, 21, 11f, 22, 11.5f, 23, 12f, 24, 12.5f, 25, 13f, 26, 13.5f, 27, 14f, 28, 14.5f, 29, 15f, 30, 16f, 32, 15.5f, 31 }, 2);

	// max pooling
	ConnectionCalculator calc = new AparapiAveragePooling2D(c, 2);
	TreeMap<Connections, Matrix> map = new TreeMap<Connections, Matrix>();
	map.put(c, a1);
	Matrix o = new Matrix(8, 2);
	calc.calculate(map, o, c.getOutputLayer());

	Map<Layer, Matrix> activations = new HashMap<Layer, Matrix>();
	activations.put(c.getInputLayer(), a1);

	BackpropagationAveragePooling2D bp = new BackpropagationAveragePooling2D(c, 2);
	bp.setActivations(activations);

	Matrix bpo = new Matrix(32, 2);
	map.put(c, o);
	bp.calculate(map, bpo, c.getInputLayer());

	assertEquals(true, bpo.get(0, 0) == o.get(0, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(1, 0) == o.get(0, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(4, 0) == o.get(0, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(5, 0) == o.get(0, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(10, 0) == o.get(3, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(11, 0) == o.get(3, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(14, 0) == o.get(3, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(15, 0) == o.get(3, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(0, 1) == o.get(0, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(1, 1) == o.get(0, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(4, 1) == o.get(0, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(5, 1) == o.get(0, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(10, 1) == o.get(3, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(11, 1) == o.get(3, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(14, 1) == o.get(3, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(15, 1) == o.get(3, 1) / c.getSubsamplingRegionLength());
    }

    @Test
    public void testCNNConstruction() {
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 32, 32 }, { 5, 5, 6 }, { 2, 2 }, { 5, 5, 16 }, { 2, 2 }, { 5, 5, 120 }, {84}, {10} }, true);
	assertEquals(13, nn.getLayers().size(), 0);

	ConvGridLayer l = (ConvGridLayer) nn.getInputLayer().getConnections().get(0).getOutputLayer();
	assertEquals(28, l.getFeatureMapRows(), 0);
	assertEquals(28, l.getFeatureMapColumns(), 0);
	assertEquals(6, l.getFilters(), 0);

	l = (ConvGridLayer) l.getConnections().get(2).getOutputLayer();
	assertEquals(14, l.getFeatureMapRows(), 0);
	assertEquals(14, l.getFeatureMapColumns(), 0);
	assertEquals(6, l.getFilters(), 0);

	l = (ConvGridLayer) l.getConnections().get(1).getOutputLayer();
	assertEquals(10, l.getFeatureMapRows(), 0);
	assertEquals(10, l.getFeatureMapColumns(), 0);
	assertEquals(16, l.getFilters(), 0);

	l = (ConvGridLayer) l.getConnections().get(2).getOutputLayer();
	assertEquals(5, l.getFeatureMapRows(), 0);
	assertEquals(5, l.getFeatureMapColumns(), 0);
	assertEquals(16, l.getFilters(), 0);

	l = (ConvGridLayer) l.getConnections().get(1).getOutputLayer();
	assertEquals(1, l.getFeatureMapRows(), 0);
	assertEquals(1, l.getFeatureMapColumns(), 0);
	assertEquals(120, l.getFilters(), 0);
	
	Layer layer = l.getConnections().get(2).getOutputLayer();
	assertEquals(84, layer.getNeuronCount(), 0);
	
	layer = layer.getConnections().get(2).getOutputLayer();
	assertEquals(10, layer.getNeuronCount(), 0);
    }
}
