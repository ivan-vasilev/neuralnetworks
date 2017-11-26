package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.RepeaterConnection;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.tensor.Tensor.TensorIterator;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.training.backpropagation.BackPropagationLayerCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

/**
 * Tests for convolutional networks
 */
@RunWith(Parameterized.class)
public class CNNTest
{
	public CNNTest(RuntimeConfiguration conf)
	{
		Environment.getInstance().setRuntimeConfiguration(conf);
	}

	@Parameters
	public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> configurations = new ArrayList<>();

		RuntimeConfiguration conf1 = new RuntimeConfiguration();
		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);
		configurations.add(new RuntimeConfiguration[] { conf1 });

		RuntimeConfiguration conf2 = new RuntimeConfiguration();
		conf2.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		conf2.setUseDataSharedMemory(true);
		conf2.setUseWeightsSharedMemory(true);
		configurations.add(new RuntimeConfiguration[] { conf2 });

		RuntimeConfiguration conf3 = new RuntimeConfiguration();
		conf3.setCalculationProvider(CalculationProvider.OPENCL);
		conf3.setUseDataSharedMemory(false);
		conf3.setUseWeightsSharedMemory(false);
		conf3.getOpenCLConfiguration().setAggregateOperations(false);
		conf3.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
		conf3.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		configurations.add(new RuntimeConfiguration[] { conf3 });

		return configurations;
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testDimensions()
	{
		// convolution dimensions
		ConnectionFactory cf = new ConnectionFactory();
		Conv2DConnection conv = cf.conv2d(new Layer(), new Layer(), 4, 4, 3, 2, 2, 2, 1, 1, 1, 1);

		assertEquals(3, conv.getOutputFeatureMapColumns(), 0);
		assertEquals(3, conv.getOutputFeatureMapRows(), 0);
		assertEquals(9, conv.getOutputFeatureMapLength(), 0);
		assertEquals(5, conv.getOutputFeatureMapColumnsWithPadding(), 0);
		assertEquals(5, conv.getOutputFeatureMapRowsWithPadding(), 0);
		assertEquals(25, conv.getOutputFeatureMapLengthWithPadding(), 0);
		assertEquals(2, conv.getOutputFilters(), 0);

		// subsampling dimensions
		Subsampling2DConnection sub = cf.subsampling2D(new Layer(), new Layer(), 5, 5, 2, 2, 3, 2, 2, 1, 1);

		assertEquals(2, sub.getOutputFeatureMapColumns(), 0);
		assertEquals(2, sub.getOutputFeatureMapRows(), 0);
		assertEquals(4, sub.getOutputFeatureMapLength(), 0);
		assertEquals(4, sub.getOutputFeatureMapColumnsWithPadding(), 0);
		assertEquals(4, sub.getOutputFeatureMapRowsWithPadding(), 0);
		assertEquals(16, sub.getOutputFeatureMapLengthWithPadding(), 0);
		assertEquals(3, sub.getFilters(), 0);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testCNNConstruction()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 32, 32, 1 }, { 5, 5, 6, 1, 1, 0, 0 }, { 2, 2, 2, 2, 0, 0 }, { 5, 5, 16, 1, 1, 0, 0 }, { 2, 2, 2, 2, 0, 0 }, { 5, 5, 120, 1, 1, 0, 0 }, { 84 }, { 10 } }, true);
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

	@Ignore
	@Test
	@Deprecated // test moved
	public void testCNNConstruction2()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 5, 5, 20, 1, 1, 0, 0 }, { 2, 2, 2, 2, 0, 0 }, { 5, 5, 50, 1, 1, 0, 0 }, { 2, 2, 1, 2, 0, 0 }, { 500 }, { 10 } }, true);
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
		assertEquals(7, sc.getOutputFeatureMapRows(), 0);
		assertEquals(4, sc.getOutputFeatureMapColumns(), 0);
		assertEquals(50, sc.getFilters(), 0);
		assertEquals(50 * 7 * 4, l.getConnections().get(0).getOutputUnitCount(), 0);

		Layer layer = l.getConnections().get(1).getOutputLayer();
		assertEquals(500, layer.getConnections().get(0).getOutputUnitCount(), 0);

		layer = layer.getConnections().get(2).getOutputLayer();
		assertEquals(500, layer.getConnections().get(0).getInputUnitCount(), 0);
		assertEquals(10, layer.getConnections().get(0).getOutputUnitCount(), 0);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testCNNConstruction3()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 6, 6, 1 }, { 3, 3, 2, 2, 2, 2, 2 }, { 2, 2, 2, 1, 1, 1 }, { 11 } }, true);
		assertEquals(6, nn.getLayers().size(), 0);

		Conv2DConnection cc = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
		Layer l = nn.getInputLayer().getConnections().get(0).getOutputLayer();
		assertEquals(2, cc.getOutputFeatureMapRows(), 0);
		assertEquals(2, cc.getOutputFeatureMapColumns(), 0);
		assertEquals(6, cc.getOutputFeatureMapRowsWithPadding(), 0);
		assertEquals(6, cc.getOutputFeatureMapColumnsWithPadding(), 0);
		assertEquals(2, cc.getOutputFilters(), 0);

		Subsampling2DConnection sc = (Subsampling2DConnection) l.getConnections().get(2);
		l = l.getConnections().get(2).getOutputLayer();
		assertEquals(6, sc.getInputFeatureMapRows(), 0);
		assertEquals(6, sc.getInputFeatureMapColumns(), 0);
		assertEquals(3, sc.getOutputFeatureMapRows(), 0);
		assertEquals(5, sc.getOutputFeatureMapColumns(), 0);
		assertEquals(5, sc.getOutputFeatureMapRowsWithPadding(), 0);
		assertEquals(7, sc.getOutputFeatureMapColumnsWithPadding(), 0);
		assertEquals(2, sc.getFilters(), 0);

		FullyConnected fc = (FullyConnected) nn.getOutputLayer().getConnections().get(0);
		assertEquals(70, fc.getInputUnitCount(), 0);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testCNNLayerCalculatorConstruction()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 5, 5, 20, 1, 1, 0, 0 }, { 2, 2, 2, 2, 0, 0 }, { 5, 5, 50, 1, 1, 0, 0 }, { 2, 2, 2, 2, 0, 0 }, { 500 }, { 10 } }, true);
		nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, null));
		CalculationFactory.lcMaxPooling(nn);

		// feedforwad cc
		LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();

		Layer l = nn.getInputLayer();

		assertTrue(OperationsFactory.isConv2D(lc.getConnectionCalculator(l)));

		l = l.getConnections().get(0).getOutputLayer();
		assertTrue(OperationsFactory.isConv2D(lc.getConnectionCalculator(l)));

		l = l.getConnections().get(2).getOutputLayer();
		assertTrue(OperationsFactory.isMaxPooling2D(lc.getConnectionCalculator(l)));

		l = l.getConnections().get(1).getOutputLayer();
		assertTrue(OperationsFactory.isConv2D(lc.getConnectionCalculator(l)));

		l = l.getConnections().get(2).getOutputLayer();
		assertTrue(OperationsFactory.isMaxPooling2D(lc.getConnectionCalculator(l)));

		l = l.getConnections().get(1).getOutputLayer();
		assertTrue(OperationsFactory.isSigmoidConnectionCalculator(lc.getConnectionCalculator(l)));

		l = l.getConnections().get(2).getOutputLayer();
		assertTrue(OperationsFactory.isSigmoidConnectionCalculator(lc.getConnectionCalculator(l)));

		// backpropagation cc
		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, null, null, null, null, 0.01f, 0.5f, 0f, 0f, 0f, 1, 1, 1);
		BackPropagationLayerCalculatorImpl bplc = (BackPropagationLayerCalculatorImpl) bpt.getBPLayerCalculator();

		l = nn.getInputLayer();
		assertTrue(OperationsFactory.isBPSigmoid((BackPropagationConnectionCalculator) bplc.getConnectionCalculator(l)));

		l = l.getConnections().get(0).getOutputLayer();
		assertTrue(OperationsFactory.isBPMaxPooling2D((BackPropagationConnectionCalculator) bplc.getConnectionCalculator(l)));
		assertTrue(OperationsFactory.isBPSigmoid((BackPropagationConnectionCalculator) bplc.getConnectionCalculator(l.getConnections().get(0).getInputLayer())));
		assertFalse(OperationsFactory.isBPSigmoid((BackPropagationConnectionCalculator) bplc.getConnectionCalculator(l.getConnections().get(1).getInputLayer()))); // bias

		l = l.getConnections().get(2).getOutputLayer();
		assertFalse(OperationsFactory.isBPSigmoid((BackPropagationConnectionCalculator) bplc.getConnectionCalculator(l)));

		l = l.getConnections().get(1).getOutputLayer();
		assertTrue(OperationsFactory.isBPMaxPooling2D((BackPropagationConnectionCalculator) bplc.getConnectionCalculator(l)));
		assertFalse(OperationsFactory.isBPSigmoid((BackPropagationConnectionCalculator) bplc.getConnectionCalculator(l.getConnections().get(0).getInputLayer())));
		assertFalse(OperationsFactory.isBPSigmoid((BackPropagationConnectionCalculator) bplc.getConnectionCalculator(l.getConnections().get(1).getInputLayer()))); // bias

		l = l.getConnections().get(2).getOutputLayer();
		assertFalse(OperationsFactory.isBPSigmoid((BackPropagationConnectionCalculator) bplc.getConnectionCalculator(l)));

		l = l.getConnections().get(1).getOutputLayer();
		assertFalse(OperationsFactory.isBPSigmoid((BackPropagationConnectionCalculator) bplc.getConnectionCalculator(l.getConnections().get(1).getInputLayer())));
		assertTrue(OperationsFactory.isBPSigmoid((BackPropagationConnectionCalculator) bplc.getConnectionCalculator(l)));

		l = l.getConnections().get(2).getOutputLayer();
		assertFalse(OperationsFactory.isBPSigmoid((BackPropagationConnectionCalculator) bplc.getConnectionCalculator(l.getConnections().get(1).getInputLayer())));
		assertTrue(bplc.getConnectionCalculator(l) == null);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testConvolutions()
	{
		Conv2DConnection c = new ConnectionFactory().conv2d(new Layer(), new Layer(), 3, 3, 2, 2, 2, 1, 1, 1, 2, 1);

		c.getWeights().setElements(new float[] { 1, 2, 3, 4, 1, 2, 3, 4 });

		ValuesProvider vp = TensorFactory.tensorProvider(c, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		TensorIterator it = vp.get(c.getInputLayer()).iterator();
		for (int i = 0; i < vp.get(c.getInputLayer()).getSize(); i++)
		{
			vp.get(c.getInputLayer()).getElements()[it.next()] = i + 1;
		}

		ConnectionCalculator conv =  OperationsFactory.conv2D();
		List<Connections> connections = new ArrayList<>();
		connections.add(c);
		conv.calculate(connections, vp, c.getOutputLayer());

		// most simple case
		Tensor o = vp.get(c.getOutputLayer(), 1, 1, 2, 2);

		Tensor oPadding = vp.get(c.getOutputLayer(), 1, 1, 6, 4);
		assertTrue(o.getElements() == oPadding.getElements());

		assertEquals(164, o.get(0, 0, 0, 0), 0);
		assertEquals(184, o.get(0, 0, 0, 1), 0);
		assertEquals(224, o.get(0, 0, 1, 0), 0);
		assertEquals(244, o.get(0, 0, 1, 1), 0);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testConvolutionsStride()
	{
		Conv2DConnection c = new ConnectionFactory().conv2d(new Layer(), new Layer(), 4, 4, 1, 2, 2, 1, 1, 2, 0, 0);

		c.getWeights().forEach(i -> c.getWeights().getElements()[i] = 1);

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15 };
		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);

		ConnectionCalculator conv =  OperationsFactory.conv2D();
		List<Connections> connections = new ArrayList<>();
		connections.add(c);
		conv.calculate(connections, vp, c.getOutputLayer());

		// most simple case
		Tensor o = vp.get(c.getOutputLayer());

		assertEquals(3, o.getDimensions()[2]);
		assertEquals(2, o.getDimensions()[3]);

		assertEquals(7, o.get(0, 0, 0, 0), 0);
		assertEquals(11, o.get(0, 0, 0, 1), 0);
		assertEquals(15, o.get(0, 0, 1, 0), 0);
		assertEquals(19, o.get(0, 0, 1, 1), 0);
		assertEquals(23, o.get(0, 0, 2, 0), 0);
		assertEquals(27, o.get(0, 0, 2, 1), 0);

		assertEquals(14, o.get(1, 0, 0, 0), 0);
		assertEquals(22, o.get(1, 0, 0, 1), 0);
		assertEquals(30, o.get(1, 0, 1, 0), 0);
		assertEquals(38, o.get(1, 0, 1, 1), 0);
		assertEquals(46, o.get(1, 0, 2, 0), 0);
		assertEquals(54, o.get(1, 0, 2, 1), 0);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testConvolutionsStride2()
	{
		Conv2DConnection c = new ConnectionFactory().conv2d(new Layer(), new Layer(), 4, 4, 1, 2, 2, 1, 2, 2, 0, 0);
		
		c.getWeights().forEach(i -> c.getWeights().getElements()[i] = 1);
		
		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15 };
		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);
		
		ConnectionCalculator conv =  OperationsFactory.conv2D();
		List<Connections> connections = new ArrayList<>();
		connections.add(c);
		conv.calculate(connections, vp, c.getOutputLayer());
		
		// most simple case
		Tensor o = vp.get(c.getOutputLayer());
		
		assertEquals(2, o.getDimensions()[2]);
		assertEquals(2, o.getDimensions()[3]);
		
		assertEquals(7, o.get(0, 0, 0, 0), 0);
		assertEquals(11, o.get(0, 0, 0, 1), 0);
		assertEquals(23, o.get(0, 0, 1, 0), 0);
		assertEquals(27, o.get(0, 0, 1, 1), 0);
		
		assertEquals(14, o.get(1, 0, 0, 0), 0);
		assertEquals(22, o.get(1, 0, 0, 1), 0);
		assertEquals(46, o.get(1, 0, 1, 0), 0);
		assertEquals(54, o.get(1, 0, 1, 1), 0);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testConvolutions2()
	{
		Conv2DConnection c = new ConnectionFactory().conv2d(new Layer(), new Layer(), 3, 3, 2, 2, 2, 2, 1, 1, 0, 0);
		c.getWeights().setElements(new float[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 });

		ValuesProvider vp = TensorFactory.tensorProvider(c, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		TensorIterator it = vp.get(c.getInputLayer()).iterator();
		for (int i = 0; i < vp.get(c.getInputLayer()).getSize(); i++)
		{
			vp.get(c.getInputLayer()).getElements()[it.next()] = i + 1;
		}

		ConnectionCalculator conv =  OperationsFactory.conv2D();
		List<Connections> connections = new ArrayList<>();
		connections.add(c);
		conv.calculate(connections, vp, c.getOutputLayer());

		Tensor o = vp.get(c.getOutputLayer());

		assertEquals(164, o.get(0, 0, 0, 0), 0);
		assertEquals(184, o.get(0, 0, 0, 1), 0);
		assertEquals(224, o.get(0, 0, 1, 0), 0);
		assertEquals(244, o.get(0, 0, 1, 1), 0);
		assertEquals(164, o.get(0, 1, 0, 0), 0);
		assertEquals(184, o.get(0, 1, 0, 1), 0);
		assertEquals(224, o.get(0, 1, 1, 0), 0);
		assertEquals(244, o.get(0, 1, 1, 1), 0);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testConvolutions3()
	{
		Conv2DConnection c = new ConnectionFactory().conv2d(new Layer(), new Layer(), 2, 2, 2, 2, 2, 2, 1, 1, 2, 3);
		TensorIterator it = c.getWeights().iterator();
		for (int i = 1; it.hasNext(); i++) {
			c.getWeights().getElements()[it.next()] = i * 0.5f;
		}

		Conv2DConnection b = new ConnectionFactory().conv2d(new Layer(), c.getOutputLayer(), 1, 1, 1, 1, 1, 2, 1, 1, 2, 3);
		it = b.getWeights().iterator();
		for (int i = 1; it.hasNext(); i++) {
			b.getWeights().getElements()[it.next()] = i * 0.5f;
		}

		NeuralNetworkImpl nn = new NeuralNetworkImpl();
		nn.addConnections(c, b);

		ValuesProvider vp = TensorFactory.tensorProvider(nn, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

		it = vp.get(c.getInputLayer()).iterator(new int[][] { {0, 0, 0, 0 }, {0, 1, 1, 1}});
		for (int i = 1; it.hasNext(); i++)
		{
			vp.get(c.getInputLayer()).getElements()[it.next()] = i;
		}
		
		it = vp.get(c.getInputLayer()).iterator(new int[][] { {1, 0, 0, 0 }, {1, 1, 1, 1}});
		for (int i = 1; it.hasNext(); i++)
		{
			vp.get(c.getInputLayer()).getElements()[it.next()] = i;
		}

		vp.get(b.getInputLayer()).forEach(i -> vp.get(b.getInputLayer()).getElements()[i] = 1);

		ConnectionCalculator conv =  OperationsFactory.conv2D();
		List<Connections> connections = new ArrayList<>();
		connections.add(c);
		connections.add(b);
		conv.calculate(connections, vp, c.getOutputLayer());
		
		Tensor o = TensorFactory.tensor(c.getOutputLayer(), c, vp);

		assertEquals(2, o.getDimensions()[0], 0);
		assertEquals(2, o.getDimensions()[1], 0);
		assertEquals(1, o.getDimensions()[2], 0);
		assertEquals(1, o.getDimensions()[3], 0);
		assertEquals(140, o.getElements().length - o.getStartOffset());

		assertEquals(102.5, o.get(0, 0, 0, 0), 0);
		assertEquals(247, o.get(0, 1, 0, 0), 0);
		assertEquals(102.5, o.get(1, 0, 0, 0), 0);
		assertEquals(247, o.get(1, 1, 0, 0), 0);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testSimpleCNN()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 3, 3, 2 }, { 2, 2, 2, 1, 1, 0, 0 }, { 2, 2, 2, 2, 0, 0 } }, false);
		nn.setLayerCalculator(CalculationFactory.lcWeightedSum(nn, null));
		CalculationFactory.lcMaxPooling(nn);

		Conv2DConnection c = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
		c.getWeights().setElements(new float[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 });

		ValuesProvider vp = TensorFactory.tensorProvider(nn, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		TensorIterator it = vp.get(c.getInputLayer()).iterator();
		for (int i = 0; i < vp.get(c.getInputLayer()).getSize(); i++)
		{
			vp.get(c.getInputLayer()).getElements()[it.next()] = i + 1;
		}

		Set<Layer> calculatedLayers = new HashSet<>();
		calculatedLayers.add(nn.getInputLayer());
		nn.getLayerCalculator().calculate(nn, nn.getOutputLayer(), calculatedLayers, vp);

		Tensor o = vp.get(nn.getOutputLayer());

		assertEquals(244, o.get(0, 0, 0, 0), 0);
		assertEquals(244, o.get(0, 1, 0, 0), 0);
	}

	@Ignore
	@Test
	@Deprecated // test moved to OpenCLMaxPooling2DTest
	public void testMaxPooling()
	{
		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 2, 2, 2, 2, 1);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ConnectionCalculator calc = OperationsFactory.maxPooling2D();

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 8.5f, 9f, 9.5f, 10f, 10.5f, 11f, 11.5f, 12f, 12.5f, 13f, 13.5f, 14f, 14.5f, 15f, 16f, 15.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31 };
		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);

		calc.calculate(connections, vp, c.getOutputLayer());

		Tensor o = vp.get(c.getOutputLayer(), 2, 2, 2, 2);

		// test padding
		Tensor oPadding = vp.get(c.getOutputLayer(), 2, 2, 6, 4);
		assertTrue(o.getElements() == oPadding.getElements());

		// test values
		assertEquals(3, o.get(0, 0, 0, 0), 0);
		assertEquals(4, o.get(0, 0, 0, 1), 0);
		assertEquals(7, o.get(0, 0, 1, 0), 0);
		assertEquals(8, o.get(0, 0, 1, 1), 0);
		assertEquals(11, o.get(0, 1, 0, 0), 0);
		assertEquals(12, o.get(0, 1, 0, 1), 0);
		assertEquals(15, o.get(0, 1, 1, 0), 0);
		assertEquals(16, o.get(0, 1, 1, 1), 0);

		assertEquals(6, o.get(1, 0, 0, 0), 0);
		assertEquals(8, o.get(1, 0, 0, 1), 0);
		assertEquals(14, o.get(1, 0, 1, 0), 0);
		assertEquals(16, o.get(1, 0, 1, 1), 0);
		assertEquals(22, o.get(1, 1, 0, 0), 0);
		assertEquals(24, o.get(1, 1, 0, 1), 0);
		assertEquals(30, o.get(1, 1, 1, 0), 0);
		assertEquals(32, o.get(1, 1, 1, 1), 0);
	}

	@Ignore
	@Test
	@Deprecated // moved to OpenCLMaxPooling2DTest
	public void testSubsamplingStride()
	{
		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 2, 2, 1, 0, 0);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ConnectionCalculator calc = OperationsFactory.maxPooling2D();

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 8.5f, 9f, 9.5f, 10f, 10.5f, 11f, 11.5f, 12f, 12.5f, 13f, 13.5f, 14f, 14.5f, 15f, 16f, 15.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31 };
		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);

		calc.calculate(connections, vp, c.getOutputLayer());

		Tensor o = vp.get(c.getOutputLayer());
		assertEquals(2, o.getDimensions()[2]);
		assertEquals(3, o.getDimensions()[3]);
		
		assertEquals(3, o.get(0, 0, 0, 0), 0);
		assertEquals(3.5, o.get(0, 0, 0, 1), 0);
		assertEquals(4, o.get(0, 0, 0, 2), 0);
		assertEquals(7, o.get(0, 0, 1, 0), 0);
		assertEquals(8, o.get(0, 0, 1, 1), 0);
		assertEquals(8, o.get(0, 0, 1, 2), 0);

		assertEquals(6, o.get(1, 0, 0, 0), 0);
		assertEquals(7, o.get(1, 0, 0, 1), 0);
		assertEquals(8, o.get(1, 0, 0, 2), 0);
		assertEquals(14, o.get(1, 0, 1, 0), 0);
		assertEquals(16, o.get(1, 0, 1, 1), 0);
		assertEquals(16, o.get(1, 0, 1, 2), 0);
		
		assertEquals(11, o.get(0, 1, 0, 0), 0);
		assertEquals(11.5, o.get(0, 1, 0, 1), 0);
		assertEquals(12, o.get(0, 1, 0, 2), 0);
		assertEquals(15, o.get(0, 1, 1, 0), 0);
		assertEquals(16, o.get(0, 1, 1, 1), 0);
		assertEquals(16, o.get(0, 1, 1, 2), 0);
		
		assertEquals(22, o.get(1, 1, 0, 0), 0);
		assertEquals(23, o.get(1, 1, 0, 1), 0);
		assertEquals(24, o.get(1, 1, 0, 2), 0);
		assertEquals(30, o.get(1, 1, 1, 0), 0);
		assertEquals(32, o.get(1, 1, 1, 1), 0);
		assertEquals(32, o.get(1, 1, 1, 2), 0);
	}

	@Ignore
	@Test
	@Deprecated // moved to OpenCLMaxPooling2DTest
	public void testSubsamplingStrideBackpropagation()
	{
		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 2, 2, 1, 0, 0);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		// max pooling
		ValuesProvider activations = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 8.5f, 9f, 9.5f, 10f, 10.5f, 11f, 11.5f, 12f, 12.5f, 13f, 13.5f, 14f, 14.5f, 15f, 16f, 15.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31 };
		System.arraycopy(src, 0, activations.get(c.getInputLayer()).getElements(), activations.get(c.getInputLayer()).getStartIndex(), src.length);

		ConnectionCalculator calc = OperationsFactory.maxPooling2D();
		calc.calculate(connections, activations, c.getOutputLayer());

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		TensorFactory.copy(activations.get(c.getOutputLayer()), vp.get(c.getOutputLayer()));

		BackPropagationConnectionCalculator bp = OperationsFactory.bpMaxPooling(new Properties());
		bp.setActivations(activations);
		bp.calculate(connections, vp, c.getInputLayer());

		Tensor a = activations.get(c.getInputLayer());
		Tensor bpo = vp.get(c.getInputLayer());

		assertEquals(false, bpo.get(0, 0, 0, 1) >= a.get(0, 0, 0, 1));
		assertEquals(true, bpo.get(0, 0, 1, 1) >= a.get(0, 0, 1, 1));
		assertEquals(true, bpo.get(0, 0, 1, 2) >= a.get(0, 0, 1, 2));
		assertEquals(false, bpo.get(0, 0, 3, 3) >= a.get(0, 0, 3, 3));
		assertEquals(true, bpo.get(0, 0, 3, 1) >= a.get(0, 0, 3, 1));
		assertEquals(true, bpo.get(0, 0, 3, 2) >= a.get(0, 0, 3, 2));

		assertEquals(false, bpo.get(0, 1, 0, 1) >= a.get(0, 1, 0, 1));
		assertEquals(true, bpo.get(0, 1, 1, 1) >= a.get(0, 1, 1, 1));
		assertEquals(true, bpo.get(0, 1, 1, 2) >= a.get(0, 1, 1, 2));
		assertEquals(false, bpo.get(0, 1, 3, 3) >= a.get(0, 1, 3, 3));
		assertEquals(true, bpo.get(0, 1, 3, 1) >= a.get(0, 1, 3, 1));
		assertEquals(true, bpo.get(0, 1, 3, 2) >= a.get(0, 1, 3, 2));

		assertEquals(false, bpo.get(1, 0, 0, 1) >= a.get(1, 0, 0, 1));
		assertEquals(true, bpo.get(1, 0, 1, 1) >= a.get(1, 0, 1, 1));
		assertEquals(true, bpo.get(1, 0, 1, 2) >= a.get(1, 0, 1, 2));
		assertEquals(false, bpo.get(1, 0, 3, 3) >= a.get(1, 0, 3, 3));
		assertEquals(true, bpo.get(1, 0, 3, 1) >= a.get(1, 0, 3, 1));
		assertEquals(true, bpo.get(1, 0, 3, 2) >= a.get(1, 0, 3, 2));
		
		assertEquals(false, bpo.get(1, 1, 0, 1) >= a.get(1, 1, 0, 1));
		assertEquals(true, bpo.get(1, 1, 1, 1) >= a.get(1, 1, 1, 1));
		assertEquals(true, bpo.get(1, 1, 1, 2) >= a.get(1, 1, 1, 2));
		assertEquals(false, bpo.get(1, 1, 3, 3) >= a.get(1, 1, 3, 3));
		assertEquals(true, bpo.get(1, 1, 3, 1) >= a.get(1, 1, 3, 1));
		assertEquals(true, bpo.get(1, 1, 3, 2) >= a.get(1, 1, 3, 2));
	}

	@Ignore
	@Test
	@Deprecated // moved to OpenCLAvaragePooling2DTest
	public void testAveragePooling()
	{
		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 2, 2, 2, 0, 0);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ConnectionCalculator calc = OperationsFactory.averagePooling2D();

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 8.5f, 9f, 9.5f, 10f, 10.5f, 11f, 11.5f, 12f, 12.5f, 13f, 13.5f, 14f, 14.5f, 15f, 16f, 15.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31 };
		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);

		calc.calculate(connections, vp, c.getOutputLayer());

		Tensor o = vp.get(c.getOutputLayer());

		assertEquals(1.75, o.get(0, 0, 0, 0), 0);
		assertEquals(2.75, o.get(0, 0, 0, 1), 0);
		assertEquals(5.75, o.get(0, 0, 1, 0), 0);
		assertEquals(6.75, o.get(0, 0, 1, 1), 0);
		assertEquals(9.75, o.get(0, 1, 0, 0), 0);
		assertEquals(10.75, o.get(0, 1, 0, 1), 0);
		assertEquals(13.75, o.get(0, 1, 1, 0), 0);
		assertEquals(14.75, o.get(0, 1, 1, 1), 0);

		assertEquals(3.5, o.get(1, 0, 0, 0), 0);
		assertEquals(5.5, o.get(1, 0, 0, 1), 0);
		assertEquals(11.5, o.get(1, 0, 1, 0), 0);
		assertEquals(13.5, o.get(1, 0, 1, 1), 0);
		assertEquals(19.5, o.get(1, 1, 0, 0), 0);
		assertEquals(21.5, o.get(1, 1, 0, 1), 0);
		assertEquals(27.5, o.get(1, 1, 1, 0), 0);
		assertEquals(29.5, o.get(1, 1, 1, 1), 0);
	}

	@Ignore
	@Test
	@Deprecated
	public void testStochasticPooling()
	{
		ConnectionFactory cf = new ConnectionFactory();

		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 3, 3, 3, 3, 1, 3, 3, 0, 0);
		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 1.6f, 0, 0, 0, 0, 0, 0, 0, 2.4f, 1.6f, 0, 0, 0, 0, 0, 0, 0, 2.4f };
		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);

		ConnectionCalculator calc = OperationsFactory.stochasticPooling2D();
		calc.calculate(connections, vp, c.getOutputLayer());

		Tensor t = vp.get(c.getOutputLayer());

		assertEquals(2.08, t.get(0, 0, 0, 0), 0.01);
		assertEquals(2.08, t.get(1, 0, 0, 0), 0.01);
	}

	@Ignore
	@Test
	@Deprecated // test moved to OpenCLMaxPooling2DTest
	public void testMaxPoolingBackpropagation()
	{
		ConnectionFactory cf = new ConnectionFactory();

		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 2, 2, 2, 2, 1);

		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		// max pooling
		ValuesProvider activations = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 8.5f, 9f, 9.5f, 10f, 10.5f, 11f, 11.5f, 12f, 12.5f, 13f, 13.5f, 14f, 14.5f, 15f, 16f, 15.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31 };
		System.arraycopy(src, 0, activations.get(c.getInputLayer()).getElements(), activations.get(c.getInputLayer()).getStartIndex(), src.length);

		ConnectionCalculator calc = OperationsFactory.maxPooling2D();
		calc.calculate(connections, activations, c.getOutputLayer());

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		TensorFactory.copy(activations.get(c.getOutputLayer(), 2, 2, 2, 2), vp.get(c.getOutputLayer(), 2, 2, 2, 2));

		BackPropagationConnectionCalculator bp = OperationsFactory.bpMaxPooling(new Properties());
		bp.setActivations(activations);
		bp.calculate(connections, vp, c.getInputLayer());

		Tensor a = activations.get(c.getInputLayer());
		Tensor bpo = vp.get(c.getInputLayer());

		assertEquals(true, bpo.get(0, 0, 1, 1) == a.get(0, 0, 1, 1));
		assertEquals(true, bpo.get(0, 0, 1, 3) == a.get(0, 0, 1, 3));
		assertEquals(true, bpo.get(0, 0, 3, 1) == a.get(0, 0, 3, 1));
		assertEquals(true, bpo.get(0, 0, 3, 2) == a.get(0, 0, 3, 2));
		assertEquals(true, bpo.get(0, 1, 1, 1) == a.get(0, 1, 1, 1));
		assertEquals(true, bpo.get(0, 1, 1, 3) == a.get(0, 1, 1, 3));
		assertEquals(true, bpo.get(0, 1, 3, 1) == a.get(0, 1, 3, 1));
		assertEquals(true, bpo.get(0, 1, 3, 2) == a.get(0, 1, 3, 2));
		assertEquals(true, bpo.get(1, 0, 1, 1) == a.get(1, 0, 1, 1));
		assertEquals(true, bpo.get(1, 0, 1, 3) == a.get(1, 0, 1, 3));
		assertEquals(true, bpo.get(1, 0, 3, 1) == a.get(1, 0, 3, 1));
		assertEquals(true, bpo.get(1, 0, 3, 2) == a.get(1, 0, 3, 2));
		assertEquals(true, bpo.get(1, 1, 1, 1) == a.get(1, 1, 1, 1));
		assertEquals(true, bpo.get(1, 1, 1, 3) == a.get(1, 1, 1, 3));
		assertEquals(true, bpo.get(1, 1, 3, 1) == a.get(1, 1, 3, 1));
		assertEquals(true, bpo.get(1, 1, 3, 2) == a.get(1, 1, 3, 2));
	}

	@Ignore
	@Test
	@Deprecated // moved to OpenCLAvaragePooling2DTest
	public void testAveragePoolingBackpropagation()
	{
		ConnectionFactory cf = new ConnectionFactory();
		Subsampling2DConnection c = cf.subsampling2D(new Layer(), new Layer(), 4, 4, 2, 2, 2, 2, 2, 0, 0);

		// average pooling
		ConnectionCalculator calc = OperationsFactory.averagePooling2D();

		List<Connections> connections = new ArrayList<Connections>();
		connections.add(c);

		ValuesProvider activations = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 8.5f, 9f, 9.5f, 10f, 10.5f, 11f, 11.5f, 12f, 12.5f, 13f, 13.5f, 14f, 14.5f, 15f, 16f, 15.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31 };
		System.arraycopy(src, 0, activations.get(c.getInputLayer()).getElements(), activations.get(c.getInputLayer()).getStartIndex(), src.length);

		calc.calculate(connections, activations, c.getOutputLayer());

		BackPropagationConnectionCalculator bp = OperationsFactory.bpAveragePooling(new Properties());
		bp.setActivations(activations);

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		TensorFactory.copy(activations.get(c.getOutputLayer()), vp.get(c.getOutputLayer()));

		bp.calculate(connections, vp, c.getInputLayer());

		Tensor o = activations.get(c.getOutputLayer());
		Tensor bpo = vp.get(c.getInputLayer());
		assertEquals(o.get(0, 0, 0, 0) / c.getSubsamplingRegionLength(), bpo.get(0, 0, 0, 0), 0);
		assertEquals(o.get(0, 0, 0, 1) / c.getSubsamplingRegionLength(), bpo.get(0, 0, 0, 2), 0);
		assertEquals(o.get(0, 0, 1, 0) / c.getSubsamplingRegionLength(), bpo.get(0, 0, 2, 0), 0);
		assertEquals(o.get(0, 0, 1, 1) / c.getSubsamplingRegionLength(), bpo.get(0, 0, 2, 2), 0);
		assertEquals(o.get(0, 1, 0, 0) / c.getSubsamplingRegionLength(), bpo.get(0, 1, 0, 0), 0);
		assertEquals(o.get(0, 1, 0, 1) / c.getSubsamplingRegionLength(), bpo.get(0, 1, 0, 2), 0);
		assertEquals(o.get(0, 1, 1, 0) / c.getSubsamplingRegionLength(), bpo.get(0, 1, 2, 0), 0);
		assertEquals(o.get(0, 1, 1, 1) / c.getSubsamplingRegionLength(), bpo.get(0, 1, 2, 2), 0);
		assertEquals(o.get(1, 0, 0, 0) / c.getSubsamplingRegionLength(), bpo.get(1, 0, 0, 0), 0);
		assertEquals(o.get(1, 0, 0, 1) / c.getSubsamplingRegionLength(), bpo.get(1, 0, 0, 2), 0);
		assertEquals(o.get(1, 0, 1, 0) / c.getSubsamplingRegionLength(), bpo.get(1, 0, 2, 0), 0);
		assertEquals(o.get(1, 0, 1, 1) / c.getSubsamplingRegionLength(), bpo.get(1, 0, 2, 2), 0);
		assertEquals(o.get(1, 1, 0, 0) / c.getSubsamplingRegionLength(), bpo.get(1, 1, 0, 0), 0);
		assertEquals(o.get(1, 1, 0, 1) / c.getSubsamplingRegionLength(), bpo.get(1, 1, 0, 2), 0);
		assertEquals(o.get(1, 1, 1, 0) / c.getSubsamplingRegionLength(), bpo.get(1, 1, 2, 0), 0);
		assertEquals(o.get(1, 1, 1, 1) / c.getSubsamplingRegionLength(), bpo.get(1, 1, 2, 2), 0);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testCNNBackpropagationValues()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 3, 3, 2 }, { 2, 2, 1, 1, 1, 0, 0 } }, true);
		nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, null));

		Conv2DConnection c = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
		TensorIterator it = c.getWeights().iterator();
		float x = 0.1f;
		while (it.hasNext())
		{
			c.getWeights().getElements()[it.next()] = x;
			x += 0.1f;
		}

		Conv2DConnection b = (Conv2DConnection) nn.getOutputLayer().getConnections().get(1);
		b.getWeights().getElements()[b.getWeights().getStartIndex()] = -3f;

		SimpleInputProvider ts = new SimpleInputProvider(new float[][] { { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f } }, new float[][] { {
				1, 1, 1, 1 } });
		BackPropagationTrainer<?> trainer = TrainerFactory.backPropagation(nn, ts, null, null, null, 0.5f, 0f, 0f, 0f, 0f, 1, 1, 1);
		BackPropagationLayerCalculatorImpl bplc = (BackPropagationLayerCalculatorImpl) trainer.getBPLayerCalculator();
		bplc.setSkipEndLayers(false);
		trainer.train();

		Tensor t = trainer.getBackpropagation().get(nn.getInputLayer());

		assertEquals(0.000757, t.get(0, 0, 0, 0), 0.0001);
		assertEquals(0.003620, t.get(0, 0, 0, 1), 0.0001);
		assertEquals(0.002438, t.get(0, 0, 0, 2), 0.0001);
		assertEquals(0.006587, t.get(0, 0, 1, 0), 0.0001);
		assertEquals(0.014184, t.get(0, 0, 1, 1), 0.0001);
		assertEquals(0.006174, t.get(0, 0, 1, 2), 0.0001);
		assertEquals(0.001392, t.get(0, 0, 2, 0), 0.0001);
		assertEquals(0.002015, t.get(0, 0, 2, 1), 0.0001);
		assertEquals(0.000450, t.get(0, 0, 2, 2), 0.0001);
		assertEquals(0, t.get(0, 1, 0, 0), 0);
		assertEquals(-0.008745, t.get(0, 1, 0, 1), 0.0001);
		assertEquals(-0.008360, t.get(0, 1, 0, 2), 0.0001);
		assertEquals(-0.027274, t.get(0, 1, 1, 0), 0.0001);
		assertEquals(-0.071377, t.get(0, 1, 1, 1), 0.0001);
		assertEquals(-0.040469, t.get(0, 1, 1, 2), 0.0001);
		assertEquals(-0.014856, t.get(0, 1, 2, 0), 0.0001);
		assertEquals(-0.031471, t.get(0, 1, 2, 1), 0.0001);
		assertEquals(-0.014418, t.get(0, 1, 2, 2), 0.0001);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testCNNBackpropagation()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 3, 3, 2 }, { 2, 2, 2, 1, 1, 0, 0 } }, true);
		nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, null));

		Conv2DConnection c = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);

		TensorIterator it = c.getWeights().iterator(new int[][] { { 0, 0, 0, 0 }, { 0, 1, 1, 1 } });
		for (float i = 0.1f; it.hasNext(); i += 0.1f)
		{
			c.getWeights().getElements()[it.next()] = i;
		}

		it = c.getWeights().iterator(new int[][] { { 1, 0, 0, 0 }, { 1, 1, 1, 1 } });
		for (float i = 0.1f; it.hasNext(); i += 0.1f)
		{
			c.getWeights().getElements()[it.next()] = i;
		}

		Conv2DConnection b = (Conv2DConnection) nn.getOutputLayer().getConnections().get(1);
		b.getWeights().getElements()[b.getWeights().getStartIndex()] = -3f;
		b.getWeights().getElements()[b.getWeights().getEndIndex()] = -3f;

		SimpleInputProvider ts = new SimpleInputProvider(new float[][] { { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f } }, new float[][] { {
				1, 1, 1, 1, 1, 1, 1, 1 } });
		BackPropagationTrainer<?> t = TrainerFactory.backPropagation(nn, ts, null, null, null, 0.5f, 0f, 0f, 0f, 0f, 1, 1, 1);
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
		assertEquals(0.11756, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.22640, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.34408, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.45292, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.59712, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.70596, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.82364, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.93248, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(-2.911599, b.getWeights().getElements()[b.getWeights().getStartIndex()], 0.00001);
		assertEquals(-2.911599, b.getWeights().getElements()[b.getWeights().getEndIndex()], 0.00001);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testCNNBackpropagation2()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 2, 1, 1 }, { 1, 1, 1, 1, 2, 3 }, { 2 }, { 2 }, { 1 } }, false);
		nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, null));
		CalculationFactory.lcMaxPooling(nn);

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

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, new SimpleInputProvider(new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }), new SimpleInputProvider(
				new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }), null, null, 1f, 0f, 0f, 0f, 0f, 1, 1, 1);
		bpt.train();

		assertEquals(0.09916, cg1.get(0, 0), 0.001);
		assertEquals(0.7978, cg1.get(0, 1), 0.001);
		assertEquals(0.3972, cg1.get(1, 0), 0.01);
		assertEquals(0.5928, cg1.get(1, 1), 0.01);
		assertEquals(0.272392, cg2.get(0, 0), 0.01);
		assertEquals(0.87305, cg2.get(0, 1), 0.01);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testCNNBackpropagation3()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 3, 3, 2 }, { 2, 2, 1, 1, 1, 0, 0 } }, true);
		nn.setLayerCalculator(CalculationFactory.lcSigmoid(nn, null));

		Conv2DConnection c = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
		TensorIterator it = c.getWeights().iterator();
		float x = 0.1f;
		while (it.hasNext())
		{
			c.getWeights().getElements()[it.next()] = x;
			x += 0.1f;
		}

		Conv2DConnection b = (Conv2DConnection) nn.getOutputLayer().getConnections().get(1);
		b.getWeights().getElements()[b.getWeights().getStartIndex()] = -3f;

		SimpleInputProvider ts = new SimpleInputProvider(new float[][] { { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f },
				{ 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f } }, new float[][] { { 1, 1, 1, 1 }, { 1, 1, 1, 1 } });
		BackPropagationTrainer<?> t = TrainerFactory.backPropagation(nn, ts, null, null, null, 0.5f, 0f, 0f, 0f, 0f, 1, 1, 1);
		t.train();

		it = c.getWeights().iterator();
		assertEquals(0.12317, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.23533, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.35966, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.47182, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.63263, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.74479, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.86911, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(0.98127, c.getWeights().getElements()[it.next()], 0.00001);
		assertEquals(-2.87839, b.getWeights().getElements()[b.getWeights().getStartIndex()], 0.00001);
	}

	@Ignore
	@Test
	@Deprecated // test moved
	public void testCNNStride()
	{
		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 5, 5, 1 }, { 2, 2, 1, 2, 2, 0, 0 } }, false);
		nn.setLayerCalculator(CalculationFactory.lcWeightedSum(nn, null));

		Conv2DConnection cc = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
		cc.getWeights().forEach(i -> cc.getWeights().getElements()[i] = 1);

		ValuesProvider vp = TensorFactory.tensorProvider(nn, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
		System.arraycopy(src, 0, vp.get(nn.getInputLayer()).getElements(), vp.get(nn.getInputLayer()).getStartIndex(), src.length);

		Set<Layer> calculatedLayers = new HashSet<>();
		calculatedLayers.add(nn.getInputLayer());
		nn.getLayerCalculator().calculate(nn, nn.getOutputLayer(), calculatedLayers, vp);

		Tensor o = vp.get(nn.getOutputLayer());
		assertEquals(16, o.get(0, 0, 0, 0), 0.00001);
		assertEquals(24, o.get(0, 0, 0, 1), 0.00001);
		assertEquals(56, o.get(0, 0, 1, 0), 0.00001);
		assertEquals(64, o.get(0, 0, 1, 1), 0.00001);
	}

	@Ignore
	@Test
	@Deprecated /// test moved
	public void testCNNMLPFF()
	{
		boolean sharedMemory = Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory();
		Environment.getInstance().getRuntimeConfiguration().setUseDataSharedMemory(false);

		// CNN
		NeuralNetworkImpl cnn = NNFactory.convNN(new int[][] { { 2, 1, 1 }, { 1, 1, 1, 1, 0, 0 }, { 1 } }, false);
		cnn.setLayerCalculator(CalculationFactory.lcSigmoid(cnn, null));
		CalculationFactory.lcMaxPooling(cnn);
		FullyConnected cnnfc = (FullyConnected) cnn.getOutputLayer().getConnections().get(0);
		cnnfc.getWeights().set(0.05f, 0, 0);
		cnnfc.getWeights().set(0.08f, 0, 1);
		ValuesProvider cnnvp = TensorFactory.tensorProvider(cnn, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor cnnin = cnnvp.get(cnn.getInputLayer());
		cnnin.set(0.2f, 0, 0, 0, 0);
		cnnin.set(0.6f, 0, 0, 1, 0);

		// MLP
		NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 2, 1 }, false);
		FullyConnected mlpfc = (FullyConnected) mlp.getOutputLayer().getConnections().get(0);
		mlpfc.getWeights().set(0.05f, 0, 0);
		mlpfc.getWeights().set(0.08f, 0, 1);
		ValuesProvider mlpvp = TensorFactory.tensorProvider(mlp, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor mlpin = mlpvp.get(mlp.getInputLayer());
		mlpin.set(0.2f, 0, 0);
		mlpin.set(0.6f, 0, 1);

		// compare ff
		Set<Layer> calculated = new HashSet<>();
		calculated.add(cnn.getInputLayer());
		cnn.getLayerCalculator().calculate(cnn, cnn.getOutputLayer(), calculated, cnnvp);

		calculated = new HashSet<>();
		calculated.add(mlp.getInputLayer());
		mlp.getLayerCalculator().calculate(mlp, mlp.getOutputLayer(), calculated, mlpvp);

		assertTrue(Arrays.equals(cnnvp.get(cnn.getOutputLayer()).getElements(), mlpvp.get(mlp.getOutputLayer()).getElements()));

		Environment.getInstance().getRuntimeConfiguration().setUseDataSharedMemory(sharedMemory);
	}

	@Ignore
	@Test
	@Deprecated /// test moved to OpenCLLRNTest
 	public void testLRN()
	{
		// FF phase
		ValuesProvider vp = new ValuesProvider(Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		RepeaterConnection c = new RepeaterConnection(new Layer(), new Layer(), 20);
		vp.add(c.getInputLayer(), 2, 5, 2, 2);
		vp.add(c.getOutputLayer(), 2, 5, 2, 2);

		Tensor t = vp.get(c.getInputLayer());
		t.set(1, 0, 0, 0, 0);
		t.set(2, 0, 0, 0, 1);
		t.set(3, 0, 0, 1, 0);
		t.set(4, 0, 0, 1, 1);
		t.set(5, 0, 1, 0, 0);
		t.set(6, 0, 1, 0, 1);
		t.set(7, 0, 1, 1, 0);
		t.set(8, 0, 1, 1, 1);
		t.set(9, 0, 2, 0, 0);
		t.set(10, 0, 2, 0, 1);
		t.set(11, 0, 2, 1, 0);
		t.set(12, 0, 2, 1, 1);
		t.set(13, 0, 3, 0, 0);
		t.set(14, 0, 3, 0, 1);
		t.set(15, 0, 3, 1, 0);
		t.set(16, 0, 3, 1, 1);
		t.set(17, 0, 4, 0, 0);
		t.set(18, 0, 4, 0, 1);
		t.set(19, 0, 4, 1, 0);
		t.set(20, 0, 4, 1, 1);
		t.set(1, 1, 0, 0, 0);
		t.set(2, 1, 0, 0, 1);
		t.set(3, 1, 0, 1, 0);
		t.set(4, 1, 0, 1, 1);
		t.set(5, 1, 1, 0, 0);
		t.set(6, 1, 1, 0, 1);
		t.set(7, 1, 1, 1, 0);
		t.set(8, 1, 1, 1, 1);
		t.set(9, 1, 2, 0, 0);
		t.set(10, 1, 2, 0, 1);
		t.set(11, 1, 2, 1, 0);
		t.set(12, 1, 2, 1, 1);
		t.set(13, 1, 3, 0, 0);
		t.set(14, 1, 3, 0, 1);
		t.set(15, 1, 3, 1, 0);
		t.set(16, 1, 3, 1, 1);
		t.set(17, 1, 4, 0, 0);
		t.set(18, 1, 4, 0, 1);
		t.set(19, 1, 4, 1, 0);
		t.set(20, 1, 4, 1, 1);

		ConnectionCalculator cc = OperationsFactory.lrnConnectionCalculator(2, 5, 0.01f, 1f);
		
		cc.calculate(Arrays.asList(new Connections[] {c}), vp, c.getOutputLayer());

		Tensor o = vp.get(c.getOutputLayer());
		assertEquals(0.791557, o.get(0, 0, 1, 0), 0.0001);
		assertEquals(1.119403, o.get(0, 1, 0, 1), 0.0001);
		assertEquals(1.176471, o.get(0, 2, 0, 0), 0.0001);
		assertEquals(2.300406, o.get(0, 4, 0, 0), 0.0001);
		assertEquals(0.791557, o.get(1, 0, 1, 0), 0.0001);
		assertEquals(1.119403, o.get(1, 1, 0, 1), 0.0001);
		assertEquals(1.176471, o.get(1, 2, 0, 0), 0.0001);
		assertEquals(2.300406, o.get(1, 4, 0, 0), 0.0001);

		BackPropagationConnectionCalculator bpLRN = OperationsFactory.bpLRN(new Properties(), cc);
		ValuesProvider bpvp = new ValuesProvider(Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		bpvp.add(c.getInputLayer(), 2, 5, 2, 2);
		bpvp.add(c.getOutputLayer(), 2, 5, 2, 2);

		bpLRN.setActivations(vp);
		bpLRN.calculate(Arrays.asList(new Connections[] {c}), bpvp, c.getInputLayer());
	}

	@Ignore
	@Test
	@Deprecated /// test moved
	public void testConvolutions4()
	{
		Conv2DConnection c = new ConnectionFactory().conv2d(new Layer(), new Layer(), 3, 4, 1, 3, 3, 1, 1, 1, 0, 0);
		c.getWeights().setElements(new float[] { 1, 1, 1, 1, -8, 1, 1, 1, 1 });

		ValuesProvider vp = TensorFactory.tensorProvider(c, 1, false);
    	vp.get(c.getInputLayer()).setElements(new float[] { 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0 });

		ConnectionCalculator conv =  OperationsFactory.conv2D();
		List<Connections> connections = new ArrayList<>();
		connections.add(c);
		conv.calculate(connections, vp, c.getOutputLayer());

		// most simple case
		Tensor o = vp.get(c.getOutputLayer());
		assertEquals(0, o.get(0, 0, 0, 0), 0);
		assertEquals(-3, o.get(0, 0, 0, 1), 0);
	}
}
