package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.Test;

import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiAveragePooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DFF;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiMaxPooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiStochasticPooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSubsampling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorConv;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorFullyConnected;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConv2D;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConv2DSigmoid;
import com.github.neuralnetworks.training.backpropagation.BackPropagationLayerCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationSigmoid;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.backpropagation.BackpropagationAveragePooling2D;
import com.github.neuralnetworks.training.backpropagation.BackpropagationMaxPooling2D;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Matrix;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.Tensor.TensorIterator;
import com.github.neuralnetworks.util.TensorFactory;

/**
 * Tests for convolutional networks
 */
public class CNNTest {

    @Test
    public void testDimensions() {
	// convolution dimensions
	ConnectionFactory cf = new ConnectionFactory(true);
	Conv2DConnection conv = cf.conv2d(new Layer(), new Layer(), 4, 4, 3, 2, 2, 2, 1);

	assertEquals(3, conv.getOutputFeatureMapColumns(), 0);
	assertEquals(3, conv.getOutputFeatureMapRows(), 0);
	assertEquals(2, conv.getOutputFilters(), 0);

	// subsampling dimensions
	Subsampling2DConnection sub = cf.subsampling2D(new Layer(), new Layer(), 5, 5, 2, 2, 3);

	assertEquals(2, sub.getOutputFeatureMapColumns(), 0);
	assertEquals(2, sub.getOutputFeatureMapRows(), 0);
	assertEquals(3, sub.getFilters(), 0);
    }

    @Test
    public void testCNNConstruction() {
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 32, 32, 1 }, { 5, 5, 6, 1 }, { 2, 2 }, { 5, 5, 16, 1 }, { 2, 2 }, { 5, 5, 120, 1 }, {84}, {10} }, true, true);
	assertEquals(13, nn.getLayers().size(), 0);

	Layer l = nn.getInputLayer().getConnections().get(0).getOutputLayer();
	Conv2DConnection cc = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
	assertEquals(28, cc.getOutputFeatureMapRows(), 0);
	assertEquals(28, cc.getOutputFeatureMapColumns(), 0);
	assertEquals(6, cc.getOutputFilters(), 0);

	Subsampling2DConnection sc = (Subsampling2DConnection) l.getConnections().get(2);
	l = l.getConnections().get(2).getOutputLayer();
	assertEquals(14, sc.getOutputFeatureMapRows(), 0);
	assertEquals(14, sc.getOutputFeatureMapColumns(), 0);
	assertEquals(6, sc.getFilters(), 0);

	cc = (Conv2DConnection) l.getConnections().get(1);
	l = l.getConnections().get(1).getOutputLayer();
	assertEquals(10, cc.getOutputFeatureMapRows(), 0);
	assertEquals(10, cc.getOutputFeatureMapColumns(), 0);
	assertEquals(16, cc.getOutputFilters(), 0);

	sc = (Subsampling2DConnection) l.getConnections().get(2);
	l = l.getConnections().get(2).getOutputLayer();
	assertEquals(5, sc.getOutputFeatureMapRows(), 0);
	assertEquals(5, sc.getOutputFeatureMapColumns(), 0);
	assertEquals(16, sc.getFilters(), 0);

	cc = (Conv2DConnection) l.getConnections().get(1);
	l = l.getConnections().get(1).getOutputLayer();
	assertEquals(1, cc.getOutputFeatureMapRows(), 0);
	assertEquals(1, cc.getOutputFeatureMapColumns(), 0);
	assertEquals(120, cc.getOutputFilters(), 0);

	FullyConnected cg = (FullyConnected) l.getConnections().get(2);
	assertEquals(84, cg.getWeights().getRows(), 0);

	FullyConnected cg2 = (FullyConnected) cg.getOutputLayer().getConnections().get(2);
	assertEquals(10, cg2.getWeights().getRows(), 0);
    }

    @Test
    public void testCNNConstruction2() {
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 5, 5, 20, 1 }, { 2, 2 }, { 5, 5, 50, 1 }, { 2, 2 }, {500}, {10} }, true, true);
	assertEquals(11, nn.getLayers().size(), 0);

	Conv2DConnection cc = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
	Layer l = nn.getInputLayer().getConnections().get(0).getOutputLayer();
	assertEquals(24, cc.getOutputFeatureMapRows(), 0);
	assertEquals(24, cc.getOutputFeatureMapColumns(), 0);
	assertEquals(20, cc.getOutputFilters(), 0);

	Subsampling2DConnection sc = (Subsampling2DConnection) l.getConnections().get(2);
	l = l.getConnections().get(2).getOutputLayer();
	assertEquals(12, sc.getOutputFeatureMapRows(), 0);
	assertEquals(12, sc.getOutputFeatureMapColumns(), 0);
	assertEquals(20, sc.getFilters(), 0);

	cc = (Conv2DConnection) l.getConnections().get(1);
	l = l.getConnections().get(1).getOutputLayer();
	assertEquals(8, cc.getOutputFeatureMapRows(), 0);
	assertEquals(8, cc.getOutputFeatureMapColumns(), 0);
	assertEquals(50, cc.getOutputFilters(), 0);

	sc = (Subsampling2DConnection) l.getConnections().get(2);
	l = l.getConnections().get(2).getOutputLayer();
	assertEquals(4, sc.getOutputFeatureMapRows(), 0);
	assertEquals(4, sc.getOutputFeatureMapColumns(), 0);
	assertEquals(50, sc.getFilters(), 0);
	assertEquals(50 * 4 * 4, l.getConnections().get(0).getOutputUnitCount(), 0);

	Layer layer = l.getConnections().get(1).getOutputLayer();
	assertEquals(500, layer.getConnections().get(0).getOutputUnitCount(), 0);

	layer = layer.getConnections().get(2).getOutputLayer();
	assertEquals(500, layer.getConnections().get(0).getInputUnitCount(), 0);
	assertEquals(10, layer.getConnections().get(0).getOutputUnitCount(), 0);
    }

    @Test
    public void testCNNConstruction3() {
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 6, 6, 1 }, { 3, 3, 2, 2 }, { 2, 2 } }, true, true);
	assertEquals(4, nn.getLayers().size(), 0);

	Conv2DConnection cc = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
	Layer l = nn.getInputLayer().getConnections().get(0).getOutputLayer();
	assertEquals(2, cc.getOutputFeatureMapRows(), 0);
	assertEquals(2, cc.getOutputFeatureMapColumns(), 0);
	assertEquals(2, cc.getOutputFilters(), 0);

	Subsampling2DConnection sc = (Subsampling2DConnection) l.getConnections().get(2);
	l = l.getConnections().get(2).getOutputLayer();
	assertEquals(1, sc.getOutputFeatureMapRows(), 0);
	assertEquals(1, sc.getOutputFeatureMapColumns(), 0);
	assertEquals(2, sc.getFilters(), 0);
    }

    public void testCNNLayerCalculatorConstruction() {
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 5, 5, 20, 1 }, { 2, 2 }, { 5, 5, 50, 1 }, { 2, 2 }, {500}, {10} }, true, true);
	nn.setLayerCalculator(NNFactory.lcSigmoid(nn, null));
	NNFactory.lcMaxPooling(nn);

	// feedforwad cc
	LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();

	Layer l = nn.getInputLayer();

	assertTrue(lc.getConnectionCalculator(l) instanceof ConnectionCalculatorConv);

	l = l.getConnections().get(0).getOutputLayer();
	assertTrue(lc.getConnectionCalculator(l) instanceof ConnectionCalculatorConv);

	l = l.getConnections().get(2).getOutputLayer();
	assertTrue(lc.getConnectionCalculator(l) instanceof AparapiMaxPooling2D);

	l = l.getConnections().get(1).getOutputLayer();
	assertTrue(lc.getConnectionCalculator(l) instanceof ConnectionCalculatorConv);

	l = l.getConnections().get(2).getOutputLayer();
	assertTrue(lc.getConnectionCalculator(l) instanceof AparapiMaxPooling2D);

	l = l.getConnections().get(1).getOutputLayer();
	assertTrue(lc.getConnectionCalculator(l) instanceof ConnectionCalculatorFullyConnected);

	l = l.getConnections().get(2).getOutputLayer();
	assertTrue(lc.getConnectionCalculator(l) instanceof ConnectionCalculatorFullyConnected);

	// backpropagation cc
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, null, null, null, null, 0.01f, 0.5f, 0f, 0f);
	BackPropagationLayerCalculatorImpl bplc = (BackPropagationLayerCalculatorImpl) bpt.getBPLayerCalculator();

	l = nn.getInputLayer();
	assertTrue(bplc.getConnectionCalculator(l) instanceof BackPropagationConv2DSigmoid);

	l = l.getConnections().get(0).getOutputLayer();
	assertTrue(bplc.getConnectionCalculator(l) instanceof BackpropagationMaxPooling2D);
	assertTrue(bplc.getConnectionCalculator(l.getConnections().get(1).getInputLayer()) instanceof BackPropagationConv2D); 	// bias

	l = l.getConnections().get(2).getOutputLayer();
	assertTrue(bplc.getConnectionCalculator(l) instanceof BackPropagationConv2DSigmoid);

	l = l.getConnections().get(1).getOutputLayer();
	assertTrue(bplc.getConnectionCalculator(l) instanceof BackpropagationMaxPooling2D);
	assertTrue(bplc.getConnectionCalculator(l.getConnections().get(1).getInputLayer()) instanceof BackPropagationConv2D);

	l = l.getConnections().get(2).getOutputLayer();
	assertTrue(bplc.getConnectionCalculator(l) instanceof BackPropagationSigmoid);

	l = l.getConnections().get(1).getOutputLayer();
	assertTrue(bplc.getConnectionCalculator(l.getConnections().get(1).getInputLayer()) instanceof BackPropagationSigmoid);
	assertTrue(bplc.getConnectionCalculator(l) instanceof BackPropagationSigmoid);

	l = l.getConnections().get(2).getOutputLayer();
	assertTrue(bplc.getConnectionCalculator(l.getConnections().get(1).getInputLayer()) instanceof BackPropagationSigmoid);
	assertTrue(bplc.getConnectionCalculator(l) == null);

	// simple convolutional network
	nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 1, 1 }, {10} }, false, true);
	nn.setLayerCalculator(NNFactory.lcSigmoid(nn, null));
	NNFactory.lcMaxPooling(nn);

	l = nn.getInputLayer();
	assertTrue(lc.getConnectionCalculator(l) instanceof ConnectionCalculatorConv);

	l = l.getConnections().get(0).getOutputLayer();
	assertTrue(lc.getConnectionCalculator(l) instanceof AparapiSubsampling2D);

	l = l.getConnections().get(0).getOutputLayer();
	assertTrue(lc.getConnectionCalculator(l) instanceof AparapiSigmoid);

	bpt = TrainerFactory.backPropagation(nn, null, null, new MultipleNeuronsOutputError(), null, 0.02f, 0.5f, 0f, 0f);
	bplc = (BackPropagationLayerCalculatorImpl) bpt.getBPLayerCalculator();

	l = nn.getInputLayer();
	assertTrue(lc.getConnectionCalculator(l) instanceof BackpropagationMaxPooling2D);

	l = l.getConnections().get(0).getOutputLayer();
	assertTrue(lc.getConnectionCalculator(l) instanceof BackPropagationSigmoid);
    }

    @Test
    public void testConvolutions() {
	Conv2DConnection c = new ConnectionFactory(true).conv2d(new Layer(), new Layer(), 3, 3, 2, 2, 2, 1, 1);

	c.getWeights().setElements(new float[] {1, 2, 3, 4, 1, 2, 3, 4});

	ValuesProvider vp = Environment.getInstance().getValuesProvider(c);
	vp.replace(c.getInputLayer(), TensorFactory.tensor(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 }, 0, 2, 3, 3, 1));

	AparapiConv2D conv = new AparapiConv2DFF(c, vp, c.getOutputLayer());
	conv.calculate(c, vp, c.getOutputLayer());

	// most simple case
	Tensor o = vp.getValues(c.getOutputLayer());

	assertEquals(164, o.get(0, 0, 0, 0), 0);
	assertEquals(184, o.get(0, 0, 1, 0), 0);
	assertEquals(224, o.get(0, 1, 0, 0), 0);
	assertEquals(244, o.get(0, 1, 1, 0), 0);
    }

    @Test
    public void testConvolutions2() {
	Conv2DConnection c = new ConnectionFactory(true).conv2d(new Layer(), new Layer(), 3, 3, 2, 2, 2, 2, 1);
	c.getWeights().setElements(new float[] {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4});

	ValuesProvider vp = Environment.getInstance().getValuesProvider(c);
	vp.replace(c.getInputLayer(), TensorFactory.tensor(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 }, 0, 2, 3, 3, 1));

	AparapiConv2D conv = new AparapiConv2DFF(c, vp, c.getOutputLayer());
	conv.calculate(c, vp, c.getOutputLayer());

	Tensor o = vp.getValues(c.getOutputLayer());

	assertEquals(164, o.get(0, 0, 0, 0), 0);
	assertEquals(184, o.get(0, 0, 1, 0), 0);
	assertEquals(224, o.get(0, 1, 0, 0), 0);
	assertEquals(244, o.get(0, 1, 1, 0), 0);
	assertEquals(164, o.get(1, 0, 0, 0), 0);
	assertEquals(184, o.get(1, 0, 1, 0), 0);
	assertEquals(224, o.get(1, 1, 0, 0), 0);
	assertEquals(244, o.get(1, 1, 1, 0), 0);
    }

    @Test
    public void testSimpleCNN() {
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] {{3, 3, 2}, {2, 2, 2, 1}, {2, 2}}, false, true);
	nn.setLayerCalculator(NNFactory.lcWeightedSum(nn, null));
	NNFactory.lcMaxPooling(nn);

	Conv2DConnection c = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
	c.getWeights().setElements(new float[] {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4});

	ValuesProvider vp = Environment.getInstance().getValuesProvider(nn);
	vp.replace(c.getInputLayer(), TensorFactory.tensor(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 }, 0, 2, 3, 3, 1));

	Set<Layer> calculatedLayers = new HashSet<>();
	calculatedLayers.add(nn.getInputLayer());
	nn.getLayerCalculator().calculate(nn, nn.getOutputLayer(), calculatedLayers, vp);

	Tensor o = vp.getValues(nn.getOutputLayer());

	assertEquals(244, o.get(0, 0, 0, 0), 0);
	assertEquals(244, o.get(1, 0, 0, 0), 0);
    }

    @Test
    public void testMaxPooling() {
	Subsampling2DConnection c = new Subsampling2DConnection(new Layer(), new Layer(), 4, 4, 2, 2, 2);
	List<Connections> connections = new ArrayList<Connections>();
	connections.add(c);

	ConnectionCalculator calc = new AparapiMaxPooling2D();

	ValuesProvider vp = Environment.getInstance().getValuesProvider(c);
	vp.replace(c.getInputLayer(), TensorFactory.tensor(new float[] { 0.5f, 1, 1, 2, 1.5f, 3, 2, 4, 2.5f, 5, 3, 6, 3.5f, 7, 4f, 8, 4.5f, 9, 5f, 10, 5.5f, 11, 6f, 12, 6.5f, 13, 7f, 14, 8f, 16, 7.5f, 15, 8.5f, 17, 9f, 18, 9.5f, 19, 10f, 20, 10.5f, 21, 11f, 22, 11.5f, 23, 12f, 24, 12.5f, 25, 13f, 26, 13.5f, 27, 14f, 28, 14.5f, 29, 15f, 30, 16f, 32, 15.5f, 31 }, 0, 2, 4, 4, 2));

	calc.calculate(connections, vp, c.getOutputLayer());

	Tensor o = vp.getValues(c.getOutputLayer());

	assertEquals(3, o.get(0, 0, 0, 0), 0);
	assertEquals(4, o.get(0, 0, 1, 0), 0);
	assertEquals(7, o.get(0, 1, 0, 0), 0);
	assertEquals(8, o.get(0, 1, 1, 0), 0);
	assertEquals(11, o.get(1, 0, 0, 0), 0);
	assertEquals(12, o.get(1, 0, 1, 0), 0);
	assertEquals(15, o.get(1, 1, 0, 0), 0);
	assertEquals(16, o.get(1, 1, 1, 0), 0);

	assertEquals(6, o.get(0, 0, 0, 1), 0);
	assertEquals(8, o.get(0, 0, 1, 1), 0);
	assertEquals(14, o.get(0, 1, 0, 1), 0);
	assertEquals(16, o.get(0, 1, 1, 1), 0);
	assertEquals(22, o.get(1, 0, 0, 1), 0);
	assertEquals(24, o.get(1, 0, 1, 1), 0);
	assertEquals(30, o.get(1, 1, 0, 1), 0);
	assertEquals(32, o.get(1, 1, 1, 1), 0);
    }

    @Test
    public void testAveragePooling() {
	Subsampling2DConnection c = new Subsampling2DConnection(new Layer(), new Layer(), 4, 4, 2, 2, 2);
	List<Connections> connections = new ArrayList<Connections>();
	connections.add(c);

	AparapiAveragePooling2D calc = new AparapiAveragePooling2D();

	ValuesProvider vp = Environment.getInstance().getValuesProvider(c);
	vp.replace(c.getInputLayer(), TensorFactory.tensor(new float[] { 0.5f, 1, 1, 2, 1.5f, 3, 2, 4, 2.5f, 5, 3, 6, 3.5f, 7, 4f, 8, 4.5f, 9, 5f, 10, 5.5f, 11, 6f, 12, 6.5f, 13, 7f, 14, 8f, 16, 7.5f, 15, 8.5f, 17, 9f, 18, 9.5f, 19, 10f, 20, 10.5f, 21, 11f, 22, 11.5f, 23, 12f, 24, 12.5f, 25, 13f, 26, 13.5f, 27, 14f, 28, 14.5f, 29, 15f, 30, 16f, 32, 15.5f, 31 }, 0, 2, 4, 4, 2));

	calc.calculate(connections, vp, c.getOutputLayer());

	Tensor o = vp.getValues(c.getOutputLayer());

	assertEquals(1.75, o.get(0, 0, 0, 0), 0);
	assertEquals(2.75, o.get(0, 0, 1, 0), 0);
	assertEquals(5.75, o.get(0, 1, 0, 0), 0);
	assertEquals(6.75, o.get(0, 1, 1, 0), 0);
	assertEquals(9.75, o.get(1, 0, 0, 0), 0);
	assertEquals(10.75, o.get(1, 0, 1, 0), 0);
	assertEquals(13.75, o.get(1, 1, 0, 0), 0);
	assertEquals(14.75, o.get(1, 1, 1, 0), 0);

	assertEquals(3.5, o.get(0, 0, 0, 1), 0);
	assertEquals(5.5, o.get(0, 0, 1, 1), 0);
	assertEquals(11.5, o.get(0, 1, 0, 1), 0);
	assertEquals(13.5, o.get(0, 1, 1, 1), 0);
	assertEquals(19.5, o.get(1, 0, 0, 1), 0);
	assertEquals(21.5, o.get(1, 0, 1, 1), 0);
	assertEquals(27.5, o.get(1, 1, 0, 1), 0);
	assertEquals(29.5, o.get(1, 1, 1, 1), 0);
    }

    @Test
    public void testStochasticPooling() {
	Subsampling2DConnection c = new Subsampling2DConnection(new Layer(), new Layer(), 3, 3, 3, 3, 1);
	List<Connections> connections = new ArrayList<Connections>();
	connections.add(c);

	ValuesProvider vp = Environment.getInstance().getValuesProvider(c);
	vp.replace(c.getInputLayer(), TensorFactory.tensor(new float[] { 1.6f, 1.6f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.4f, 2.4f }, 0, 1, 3, 3, 2));

	AparapiStochasticPooling2D calc = new AparapiStochasticPooling2D();
	calc.calculate(connections, vp, c.getOutputLayer());

	Tensor t = vp.getValues(c.getOutputLayer());

	assertEquals(2.08, t.get(0, 0, 0, 0), 0.01);
	assertEquals(2.08, t.get(0, 0, 0, 1), 0.01);
    }

    @Test
    public void testMaxPoolingBackpropagation() {
	//Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);
	Subsampling2DConnection c = new Subsampling2DConnection(new Layer(), new Layer(), 4, 4, 2, 2, 2);

	List<Connections> connections = new ArrayList<Connections>();
	connections.add(c);

	// max pooling
	ValuesProvider activations = Environment.getInstance().getValuesProvider(c);
	activations.replace(c.getInputLayer(), TensorFactory.tensor(new float[] { 0.5f, 1, 1, 2, 1.5f, 3, 2, 4, 2.5f, 5, 3, 6, 3.5f, 7, 4f, 8, 4.5f, 9, 5f, 10, 5.5f, 11, 6f, 12, 6.5f, 13, 7f, 14, 8f, 16, 7.5f, 15, 8.5f, 17, 9f, 18, 9.5f, 19, 10f, 20, 10.5f, 21, 11f, 22, 11.5f, 23, 12f, 24, 12.5f, 25, 13f, 26, 13.5f, 27, 14f, 28, 14.5f, 29, 15f, 30, 16f, 32, 15.5f, 31 }, 0, 2, 4, 4, 2));

	ConnectionCalculator calc = new AparapiMaxPooling2D();
	calc.calculate(connections, activations, c.getOutputLayer());

	ValuesProvider vp = Environment.getInstance().getValuesProvider(c);
	vp.setMiniBatchSize(2);
	vp.replace(c.getOutputLayer(), activations.getValues(c.getOutputLayer()));

	BackpropagationMaxPooling2D bp = new BackpropagationMaxPooling2D();
	bp.setActivations(activations);
	bp.calculate(connections, vp, c.getInputLayer());

	Tensor a = activations.getValues(c.getInputLayer());
	Tensor bpo = vp.getValues(c.getInputLayer());

	assertEquals(true, bpo.get(0, 1, 1, 0) == a.get(0, 1, 1, 0));
	assertEquals(true, bpo.get(0, 1, 3, 0) == a.get(0, 1, 3, 0));
	assertEquals(true, bpo.get(0, 3, 1, 0) == a.get(0, 3, 1, 0));
	assertEquals(true, bpo.get(0, 3, 2, 0) == a.get(0, 3, 2, 0));
	assertEquals(true, bpo.get(0, 1, 1, 1) == a.get(0, 1, 1, 1));
	assertEquals(true, bpo.get(0, 1, 3, 1) == a.get(0, 1, 3, 1));
	assertEquals(true, bpo.get(0, 3, 1, 1) == a.get(0, 3, 1, 1));
	assertEquals(true, bpo.get(0, 3, 2, 1) == a.get(0, 3, 2, 1));
	assertEquals(true, bpo.get(1, 1, 1, 0) == a.get(1, 1, 1, 0));
	assertEquals(true, bpo.get(1, 1, 3, 0) == a.get(1, 1, 3, 0));
	assertEquals(true, bpo.get(1, 3, 1, 0) == a.get(1, 3, 1, 0));
	assertEquals(true, bpo.get(1, 3, 2, 0) == a.get(1, 3, 2, 0));
	assertEquals(true, bpo.get(1, 1, 1, 1) == a.get(1, 1, 1, 1));
	assertEquals(true, bpo.get(1, 1, 3, 1) == a.get(1, 1, 3, 1));
	assertEquals(true, bpo.get(1, 3, 1, 1) == a.get(1, 3, 1, 1));
	assertEquals(true, bpo.get(1, 3, 2, 1) == a.get(1, 3, 2, 1));
    }

    @Test
    public void testAveragePoolingBackpropagation() {
	Subsampling2DConnection c = new Subsampling2DConnection(new Layer(), new Layer(), 4, 4, 2, 2, 2);

	// average pooling
	ConnectionCalculator calc = new AparapiAveragePooling2D();

	List<Connections> connections = new ArrayList<Connections>();
	connections.add(c);

	ValuesProvider activations = Environment.getInstance().getValuesProvider(c);
	activations.replace(c.getInputLayer(), TensorFactory.tensor(new float[] { 0.5f, 1, 1, 2, 1.5f, 3, 2, 4, 2.5f, 5, 3, 6, 3.5f, 7, 4f, 8, 4.5f, 9, 5f, 10, 5.5f, 11, 6f, 12, 6.5f, 13, 7f, 14, 8f, 16, 7.5f, 15, 8.5f, 17, 9f, 18, 9.5f, 19, 10f, 20, 10.5f, 21, 11f, 22, 11.5f, 23, 12f, 24, 12.5f, 25, 13f, 26, 13.5f, 27, 14f, 28, 14.5f, 29, 15f, 30, 16f, 32, 15.5f, 31 }, 0, 2, 4, 4, 2));

	calc.calculate(connections, activations, c.getOutputLayer());

	BackpropagationAveragePooling2D bp = new BackpropagationAveragePooling2D();
	bp.setActivations(activations);

	ValuesProvider vp = Environment.getInstance().getValuesProvider(c);
	vp.setMiniBatchSize(2);
	vp.replace(c.getOutputLayer(), activations.getValues(c.getOutputLayer()));

	bp.calculate(connections, vp, c.getInputLayer());

	Tensor o = activations.getValues(c.getOutputLayer());
	Tensor bpo = vp.getValues(c.getInputLayer());
	assertEquals(true, bpo.get(0, 0, 0, 0) == o.get(0, 0, 0, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(0, 0, 2, 0) == o.get(0, 0, 1, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(0, 2, 0, 0) == o.get(0, 1, 0, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(0, 2, 2, 0) == o.get(0, 1, 1, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(1, 0, 0, 0) == o.get(1, 0, 0, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(1, 0, 2, 0) == o.get(1, 0, 1, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(1, 2, 0, 0) == o.get(1, 1, 0, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(1, 2, 2, 0) == o.get(1, 1, 1, 0) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(0, 0, 0, 1) == o.get(0, 0, 0, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(0, 0, 2, 1) == o.get(0, 0, 1, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(0, 2, 0, 1) == o.get(0, 1, 0, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(0, 2, 2, 1) == o.get(0, 1, 1, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(1, 0, 0, 1) == o.get(1, 0, 0, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(1, 0, 2, 1) == o.get(1, 0, 1, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(1, 2, 0, 1) == o.get(1, 1, 0, 1) / c.getSubsamplingRegionLength());
	assertEquals(true, bpo.get(1, 2, 2, 1) == o.get(1, 1, 1, 1) / c.getSubsamplingRegionLength());
    }

    @Test
    public void testCNNBackpropagation() {
	//Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 3, 3, 2 }, { 2, 2, 1, 1 } }, true, true);
	nn.setLayerCalculator(NNFactory.lcSigmoid(nn, null));

	Conv2DConnection c = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
	TensorIterator it = c.getWeights().iterator();
	float x = 0.1f;
	while (it.hasNext()) {
	    c.getWeights().getElements()[it.next()] = x;
	    x += 0.1f;
	}

	Conv2DConnection b = (Conv2DConnection) nn.getOutputLayer().getConnections().get(1);
	b.getWeights().getElements()[b.getWeights().getStartIndex()] = -3f;

	SimpleInputProvider ts = new SimpleInputProvider(TensorFactory.tensor(new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f }, 0, 2, 3, 3, 1), TensorFactory.tensor(new float[] { 1, 1, 1, 1 }, 0, 1, 2, 2, 1), 1, 1);
	BackPropagationTrainer<?> t = TrainerFactory.backPropagation(nn, ts, null, null, null, 0.5f, 0f, 0f, 0f);
	t.train();

	it = c.getWeights().iterator();
	assertEquals(0.11756, c.getWeights().getElements()[it.next()], 0.00001);
	assertEquals(0.22640, c.getWeights().getElements()[it.next()], 0.00001);
	assertEquals(0.34408, c.getWeights().getElements()[it.next()], 0.00001);
	assertEquals(0.45292, c.getWeights().getElements()[it.next()], 0.00001);
	assertEquals(0.59712, c.getWeights().getElements()[it.next()], 0.00001);
	assertEquals(0.70596, c.getWeights().getElements()[it.next()], 0.00001);
	assertEquals(0.82364, c.getWeights().getElements()[it.next()], 0.00001);
	assertEquals(0.93248, c.getWeights().getElements()[it.next()], 0.00001);
	assertEquals(-2.911599, b.getWeights().getElements()[b.getWeights().getStartIndex()], 0.00001);
    }

    @Test
    public void testCNNBackpropagation2() {
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { {2, 1, 1}, {1, 1}, {2}, {2}, {1} }, false, true);
	nn.setLayerCalculator(NNFactory.lcSigmoid(nn, null));
	NNFactory.lcMaxPooling(nn);

	FullyConnected c1 = (FullyConnected) nn.getInputLayer().getConnections().get(0).getOutputLayer().getConnections().get(1).getOutputLayer().getConnections().get(1);
	Matrix cg1 = c1.getWeights();
	cg1.set(0.1f, 0, 0);
	cg1.set(0.8f, 0, 1);
	cg1.set(0.4f, 1, 0);
	cg1.set(0.6f, 1, 1);

	FullyConnected c2 = (FullyConnected) nn.getOutputLayer().getConnections().iterator().next();
	Matrix cg2 = c2.getWeights();
	cg2.set(0.3f, 0, 0);
	cg2.set(0.9f, 0, 1);

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, new SimpleInputProvider(TensorFactory.matrix(new float[][] { { 0.35f, 0.9f } }), TensorFactory.matrix(new float[][] { { 0.5f } }), 1, 1), new SimpleInputProvider(TensorFactory.matrix(new float[][] { { 0.35f, 0.9f } }), TensorFactory.matrix(new float[][] { { 0.5f } }), 1, 1), null, null, 1f, 0f, 0f, 0f);
	bpt.train();

	assertEquals(0.09916, cg1.get(0, 0), 0.001);
	assertEquals(0.7978, cg1.get(0, 1), 0.001);
	assertEquals(0.3972, cg1.get(1, 0), 0.01);
	assertEquals(0.5928, cg1.get(1, 1), 0.01);
	assertEquals(0.272392, cg2.get(0, 0), 0.01);
	assertEquals(0.87305, cg2.get(0, 1), 0.01);
    }

    @Test
    public void testCNNStride() {
	//Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 5, 5, 1 }, { 2, 2, 1, 2 } }, false, true);
	nn.setLayerCalculator(NNFactory.lcWeightedSum(nn, null));

	Conv2DConnection cc = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
	cc.getWeights().forEach(i -> cc.getWeights().getElements()[i] = 1);

	ValuesProvider vp = Environment.getInstance().getValuesProvider(nn);
	vp.replace(nn.getInputLayer(), TensorFactory.tensor(new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}, 0, 1, 5, 5, 1));

	Set<Layer> calculatedLayers = new HashSet<>();
	calculatedLayers.add(nn.getInputLayer());
	nn.getLayerCalculator().calculate(nn, nn.getOutputLayer(), calculatedLayers, vp);

	Tensor o = vp.getValues(nn.getOutputLayer());
	assertEquals(16, o.get(0, 0, 0, 0), 0.00001);
	assertEquals(24, o.get(0, 0, 1, 0), 0.00001);
	assertEquals(56, o.get(0, 1, 0, 0), 0.00001);
	assertEquals(64, o.get(0, 1, 1, 0), 0.00001);
    }
}
