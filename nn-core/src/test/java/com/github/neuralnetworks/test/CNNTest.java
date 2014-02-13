package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.Test;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiAveragePooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DFF;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiMaxPooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiStochasticPooling2D;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
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
    public void testCNNConstruction() {
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 32, 32, 1 }, { 5, 5, 6 }, { 2, 2 }, { 5, 5, 16 }, { 2, 2 }, { 5, 5, 120 }, {84}, {10} }, true);
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

	GraphConnections cg = (GraphConnections) l.getConnections().get(2);
	assertEquals(84, cg.getConnectionGraph().getRows(), 0);

	GraphConnections cg2 = (GraphConnections) cg.getOutputLayer().getConnections().get(2);
	assertEquals(10, cg2.getConnectionGraph().getRows(), 0);
    }

    @Test
    public void testCNNConstruction2() {
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 5, 5, 20 }, { 2, 2 }, { 5, 5, 50 }, { 2, 2 }, {500}, {10} }, true);
	assertEquals(11, nn.getLayers().size(), 0);

	ConvGridLayer l = (ConvGridLayer) nn.getInputLayer().getConnections().get(0).getOutputLayer();
	assertEquals(24, l.getFeatureMapRows(), 0);
	assertEquals(24, l.getFeatureMapColumns(), 0);
	assertEquals(20, l.getFilters(), 0);

	l = (ConvGridLayer) l.getConnections().get(2).getOutputLayer();
	assertEquals(12, l.getFeatureMapRows(), 0);
	assertEquals(12, l.getFeatureMapColumns(), 0);
	assertEquals(20, l.getFilters(), 0);

	l = (ConvGridLayer) l.getConnections().get(1).getOutputLayer();
	assertEquals(8, l.getFeatureMapRows(), 0);
	assertEquals(8, l.getFeatureMapColumns(), 0);
	assertEquals(50, l.getFilters(), 0);

	l = (ConvGridLayer) l.getConnections().get(2).getOutputLayer();
	assertEquals(4, l.getFeatureMapRows(), 0);
	assertEquals(4, l.getFeatureMapColumns(), 0);
	assertEquals(50, l.getFilters(), 0);
	assertEquals(50 * 4 * 4, l.getConnections().get(0).getOutputUnitCount(), 0);

	Layer layer = l.getConnections().get(1).getOutputLayer();
	assertEquals(500, layer.getConnections().get(0).getOutputUnitCount(), 0);

	layer = layer.getConnections().get(2).getOutputLayer();
	assertEquals(10, layer.getConnections().get(0).getOutputUnitCount(), 0);
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
    public void testSimpleCNN() {
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] {{3, 3, 2}, {2, 2, 2}, {2, 2}}, false);
	nn.setLayerCalculator(NNFactory.lcWeightedSum(nn, null));
	NNFactory.lcMaxPooling(nn, (LayerCalculatorImpl) nn.getLayerCalculator());

	Conv2DConnection c = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
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

	ValuesProvider vp = new ValuesProvider();
	vp.addValues(nn.getInputLayer(), i1);

	Set<Layer> calculatedLayers = new HashSet<>();
	calculatedLayers.add(nn.getInputLayer());
	nn.getLayerCalculator().calculate(nn, nn.getOutputLayer(), calculatedLayers, vp);

	Matrix o = vp.getValues(nn.getOutputLayer());

	assertEquals(244, o.get(0, 0), 0);
	assertEquals(244, o.get(1, 0), 0);
    }

    @Test
    public void testMaxPooling() {
	Subsampling2DConnection c = new Subsampling2DConnection(new ConvGridLayer(4, 4, 2), 2, 2);
	Matrix i1 = new Matrix(new float[] { 0.5f, 1, 1, 2, 1.5f, 3, 2, 4, 2.5f, 5, 3, 6, 3.5f, 7, 4f, 8, 4.5f, 9, 5f, 10, 5.5f, 11, 6f, 12, 6.5f, 13, 7f, 14, 8f, 16, 7.5f, 15, 8.5f, 17, 9f, 18, 9.5f, 19, 10f, 20, 10.5f, 21, 11f, 22, 11.5f, 23, 12f, 24, 12.5f, 25, 13f, 26, 13.5f, 27, 14f, 28, 14.5f, 29, 15f, 30, 16f, 32, 15.5f, 31 }, 2);
	List<Connections> connections = new ArrayList<Connections>();
	connections.add(c);

	ConnectionCalculator calc = new AparapiMaxPooling2D();
	Matrix o = new Matrix(8, 2);

	ValuesProvider vp = new ValuesProvider();
	vp.addValues(c.getInputLayer(), i1);
	vp.addValues(c.getOutputLayer(), o);

	calc.calculate(connections, vp, c.getOutputLayer());

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
	List<Connections> connections = new ArrayList<Connections>();
	connections.add(c);

	AparapiAveragePooling2D calc = new AparapiAveragePooling2D();
	Matrix o = new Matrix(8, 2);

	ValuesProvider vp = new ValuesProvider();
	vp.addValues(c.getInputLayer(), i1);
	vp.addValues(c.getOutputLayer(), o);

	calc.calculate(connections, vp, c.getOutputLayer());

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
	List<Connections> connections = new ArrayList<Connections>();
	connections.add(c);

	Matrix o = new Matrix(1, 2);

	ValuesProvider vp = new ValuesProvider();
	vp.addValues(c.getInputLayer(), i1);
	vp.addValues(c.getOutputLayer(), o);

	AparapiStochasticPooling2D calc = new AparapiStochasticPooling2D();
	calc.calculate(connections, vp, c.getOutputLayer());

	assertEquals(2.08, o.get(0, 0), 0.01);
	assertEquals(2.08, o.get(0, 1), 0.01);
    }

    @Test
    public void testMaxPoolingBackpropagation() {
	Subsampling2DConnection c = new Subsampling2DConnection(new ConvGridLayer(4, 4, 2), 2, 2);
	Matrix a1 = new Matrix(new float[] { 0.5f, 1, 1, 2, 1.5f, 3, 2, 4, 2.5f, 5, 3, 6, 3.5f, 7, 4f, 8, 4.5f, 9, 5f, 10, 5.5f, 11, 6f, 12, 6.5f, 13, 7f, 14, 8f, 16, 7.5f, 15, 8.5f, 17, 9f, 18, 9.5f, 19, 10f, 20, 10.5f, 21, 11f, 22, 11.5f, 23, 12f, 24, 12.5f, 25, 13f, 26, 13.5f, 27, 14f, 28, 14.5f, 29, 15f, 30, 16f, 32, 15.5f, 31 }, 2);
	Matrix o = new Matrix(8, 2);

	List<Connections> connections = new ArrayList<Connections>();
	connections.add(c);

	// max pooling
	ValuesProvider vp = new ValuesProvider();
	vp.addValues(c.getInputLayer(), a1);
	vp.addValues(c.getOutputLayer(), o);

	ConnectionCalculator calc = new AparapiMaxPooling2D();
	calc.calculate(connections, vp, c.getOutputLayer());

	ValuesProvider activations = new ValuesProvider();
	activations.addValues(c.getInputLayer(), a1);

	Matrix bpo = new Matrix(32, 2);

	vp = new ValuesProvider();
	vp.addValues(c.getOutputLayer(), o);
	vp.addValues(c.getInputLayer(), bpo);

	BackpropagationMaxPooling2D bp = new BackpropagationMaxPooling2D();
	bp.setActivations(activations);
	bp.calculate(connections, vp, c.getInputLayer());

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
	ConnectionCalculator calc = new AparapiAveragePooling2D();

	List<Connections> connections = new ArrayList<Connections>();
	connections.add(c);

	ValuesProvider vp = new ValuesProvider();
	vp.addValues(c.getInputLayer(), a1);
	Matrix o = new Matrix(8, 2);
	vp.addValues(c.getOutputLayer(), o);

	calc.calculate(connections, vp, c.getOutputLayer());

	ValuesProvider activations = new ValuesProvider();
	activations.addValues(c.getInputLayer(), a1);

	BackpropagationAveragePooling2D bp = new BackpropagationAveragePooling2D();
	bp.setActivations(activations);

	vp = new ValuesProvider();
	vp.addValues(c.getOutputLayer(), o);
	Matrix bpo = new Matrix(32, 2);
	vp.addValues(c.getInputLayer(), bpo);

	bp.calculate(connections, vp, c.getInputLayer());

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
    public void testCNNBackpropagation() {
	Environment.getInstance().setExecutionStrategy(new SeqKernelExecution());

	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 3, 3, 2 }, { 2, 2, 1 } }, true);
	nn.setLayerCalculator(NNFactory.lcSigmoid(nn, null));

	Conv2DConnection c = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
	c.setWeights(new float [] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});

	Conv2DConnection b = (Conv2DConnection) nn.getOutputLayer().getConnections().get(1);
	b.setWeights(new float [] {-3f});
	
	SimpleInputProvider ts = new SimpleInputProvider(new float[][] { { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f } }, new float[][] { { 1, 1, 1, 1 } }, 1, 1);
	BackPropagationTrainer<?> t = TrainerFactory.backPropagation(nn, ts, null, null, null, 0.5f, 0f, 0f);
	t.train();

	assertEquals(0.11756, c.getWeights()[0], 0.00001);
	assertEquals(0.22640, c.getWeights()[1], 0.00001);
	assertEquals(0.34408, c.getWeights()[2], 0.00001);
	assertEquals(0.45292, c.getWeights()[3], 0.00001);
	assertEquals(0.59712, c.getWeights()[4], 0.00001);
	assertEquals(0.70596, c.getWeights()[5], 0.00001);
	assertEquals(0.82364, c.getWeights()[6], 0.00001);
	assertEquals(0.93248, c.getWeights()[7], 0.00001);
	assertEquals(-2.911599, b.getWeights()[0], 0.00001);
    }
}
