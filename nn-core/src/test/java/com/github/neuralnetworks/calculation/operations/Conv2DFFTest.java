package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
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
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.training.backpropagation.BackPropagationLayerCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Environment;

/**
 * Created by chass on 25.11.14.
 */
public class Conv2DFFTest extends AbstractTest
{

	/**
	 * Info:
	 * - disabled testing of Runtime.CPU_SEQ
	 * - comparing CPU with OpenCL results is enough
	 */

	@Test
	public void testConv2DFF()
	{
		Tensor seqResult = testConv2DFF(Runtime.CPU_SEQ);
		Tensor openclResult = testConv2DFF(Runtime.OPENCL);

		assertTrue(isEqual(seqResult, openclResult));
	}

	private Tensor testConv2DFF(Runtime runtime)
	{
		// set to > 0 to use as constant seed
		long seed = 13265498L;

		// size of the minibatch. Values [1, 256]
		int minibatchSize = 1;

		// initialize connection weights and input
		Random r = new Random();
		if (seed > 0)
		{
			r.setSeed(seed);
		}

		Layer inputLayer = new Layer();
		Layer outputLayer = new Layer();
		int inputFeatureMapRows = 3;
		int inputFeatureMapColumns = 3;
		int inputFilters = 1;
		int kernelRows = 2;
		int kernelColumns = 2;
		int outputFilters = 1;
		int rowStride = 1;
		int columnStride = 1;
		int outputRowPadding = 0;
		int outputColumnPadding = 0;
		Conv2DConnection connection = new ConnectionFactory().conv2d(
				inputLayer,
				outputLayer,
				inputFeatureMapRows,
				inputFeatureMapColumns,
				inputFilters,
				kernelRows,
				kernelColumns,
				outputFilters,
				rowStride,
				columnStride,
				outputRowPadding,
				outputColumnPadding);

		new RandomInitializerImpl(r, -1f, 1f).initialize(connection.getWeights());

		ValuesProvider vp = TensorFactory.tensorProvider(connection, minibatchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor input = vp.get(connection.getInputLayer());
		input.forEach(i -> input.getElements()[i] = r.nextFloat());

		// setup
		List<Connections> connections = new ArrayList<>();
		connections.add(connection);

		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

		ConnectionCalculator oclConv = OperationsFactory.conv2D();
		oclConv.calculate(connections, vp, connection.getOutputLayer());

		return vp.get(connection.getOutputLayer());
	}

	@Test
	public void testCNNStride()
	{
		Tensor seqResult = testCNNStride(Runtime.CPU_SEQ);
		Tensor openclResult = testCNNStride(Runtime.OPENCL);

		assertTrue(isEqual(seqResult, openclResult));
	}

	private Tensor testCNNStride(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

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

		return vp.get(nn.getOutputLayer());
	}

	@Test
	public void testCNNMLPFF()
	{
		testCNNMLPFF(Runtime.CPU_SEQ);
		testCNNMLPFF(Runtime.OPENCL);
	}

	private void testCNNMLPFF(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

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

	@Test
	public void testConvolutions4()
	{
		Tensor seqResult = testConvolutions4(Runtime.CPU_SEQ);
		Tensor openclResult = testConvolutions4(Runtime.OPENCL);

		assertTrue(isEqual(seqResult, openclResult));
	}

	private Tensor testConvolutions4(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

		Conv2DConnection c = new ConnectionFactory().conv2d(new Layer(), new Layer(), 3, 4, 1, 3, 3, 1, 1, 1, 0, 0);
		c.getWeights().setElements(new float[] { 1, 1, 1, 1, -8, 1, 1, 1, 1 });

		ValuesProvider vp = TensorFactory.tensorProvider(c, 1, false);
		vp.get(c.getInputLayer()).setElements(new float[] { 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0 });

		ConnectionCalculator conv = OperationsFactory.conv2D();
		List<Connections> connections = new ArrayList<>();
		connections.add(c);
		conv.calculate(connections, vp, c.getOutputLayer());

		// most simple case
		return vp.get(c.getOutputLayer());
	}

	@Test
	public void testSimpleCNN()
	{
		Tensor seqResult = testSimpleCNN(Runtime.CPU_SEQ);
		Tensor openclResult = testSimpleCNN(Runtime.OPENCL);

		assertTrue(isEqual(seqResult, openclResult));
	}

	private Tensor testSimpleCNN(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 3, 3, 2 }, { 2, 2, 2, 1, 1, 0, 0 }, { 2, 2, 2, 2, 0, 0 } }, false);
		nn.setLayerCalculator(CalculationFactory.lcWeightedSum(nn, null));

		// replace layer calculators for SubSampling layer with MaxPooling
		CalculationFactory.lcMaxPooling(nn);

		Conv2DConnection c = (Conv2DConnection) nn.getInputLayer().getConnections().get(0);
		c.getWeights().setElements(new float[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 });

		ValuesProvider vp = TensorFactory.tensorProvider(nn, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor.TensorIterator it = vp.get(c.getInputLayer()).iterator();
		for (int i = 0; i < vp.get(c.getInputLayer()).getSize(); i++)
		{
			vp.get(c.getInputLayer()).getElements()[it.next()] = i + 1;
		}

		Set<Layer> calculatedLayers = new HashSet<>();
		calculatedLayers.add(nn.getInputLayer());
		nn.getLayerCalculator().calculate(nn, nn.getOutputLayer(), calculatedLayers, vp);

		return vp.get(nn.getOutputLayer());
	}

	@Test
	public void testConvolutions()
	{
		testConvolutions(Runtime.CPU_SEQ);
		testConvolutions(Runtime.OPENCL);
	}

	private void testConvolutions(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

		Conv2DConnection c = new ConnectionFactory().conv2d(new Layer(), new Layer(), 3, 3, 2, 2, 2, 1, 1, 1, 2, 1);

		c.getWeights().setElements(new float[] { 1, 2, 3, 4, 1, 2, 3, 4 });

		ValuesProvider vp = TensorFactory.tensorProvider(c, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor.TensorIterator it = vp.get(c.getInputLayer()).iterator();
		for (int i = 0; i < vp.get(c.getInputLayer()).getSize(); i++)
		{
			vp.get(c.getInputLayer()).getElements()[it.next()] = i + 1;
		}

		ConnectionCalculator conv = OperationsFactory.conv2D();
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

	@Test
	public void testConvolutions2()
	{
		testConvolutions2(Runtime.CPU_SEQ);
		testConvolutions2(Runtime.OPENCL);
	}

	private void testConvolutions2(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

		Conv2DConnection c = new ConnectionFactory().conv2d(new Layer(), new Layer(), 3, 3, 2, 2, 2, 2, 1, 1, 0, 0);
		c.getWeights().setElements(new float[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 });

		ValuesProvider vp = TensorFactory.tensorProvider(c, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		Tensor.TensorIterator it = vp.get(c.getInputLayer()).iterator();
		for (int i = 0; i < vp.get(c.getInputLayer()).getSize(); i++)
		{
			vp.get(c.getInputLayer()).getElements()[it.next()] = i + 1;
		}

		ConnectionCalculator conv = OperationsFactory.conv2D();
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

	@Test
	public void testConvolutions3()
	{
		testConvolutions3(Runtime.CPU_SEQ);
		testConvolutions3(Runtime.OPENCL);
	}

	private void testConvolutions3(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

		Conv2DConnection c = new ConnectionFactory().conv2d(new Layer(), new Layer(), 2, 2, 2, 2, 2, 2, 1, 1, 2, 3);
		Tensor.TensorIterator it = c.getWeights().iterator();
		for (int i = 1; it.hasNext(); i++)
		{
			c.getWeights().getElements()[it.next()] = i * 0.5f;
		}

		Conv2DConnection b = new ConnectionFactory().conv2d(new Layer(), c.getOutputLayer(), 1, 1, 1, 1, 1, 2, 1, 1, 2, 3);
		it = b.getWeights().iterator();
		for (int i = 1; it.hasNext(); i++)
		{
			b.getWeights().getElements()[it.next()] = i * 0.5f;
		}

		NeuralNetworkImpl nn = new NeuralNetworkImpl();
		nn.addConnections(c, b);

		ValuesProvider vp = TensorFactory.tensorProvider(nn, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

		it = vp.get(c.getInputLayer()).iterator(new int[][] { { 0, 0, 0, 0 }, { 0, 1, 1, 1 } });
		for (int i = 1; it.hasNext(); i++)
		{
			vp.get(c.getInputLayer()).getElements()[it.next()] = i;
		}

		it = vp.get(c.getInputLayer()).iterator(new int[][] { { 1, 0, 0, 0 }, { 1, 1, 1, 1 } });
		for (int i = 1; it.hasNext(); i++)
		{
			vp.get(c.getInputLayer()).getElements()[it.next()] = i;
		}

		vp.get(b.getInputLayer()).forEach(i -> vp.get(b.getInputLayer()).getElements()[i] = 1);

		ConnectionCalculator conv = OperationsFactory.conv2D();
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

	@Test
	public void testConvolutionsStride()
	{
		testConvolutionsStride(Runtime.CPU_SEQ);
		testConvolutionsStride(Runtime.OPENCL);
	}

	private void testConvolutionsStride(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

		Conv2DConnection c = new ConnectionFactory().conv2d(new Layer(), new Layer(), 4, 4, 1, 2, 2, 1, 1, 2, 0, 0);

		c.getWeights().forEach(i -> c.getWeights().getElements()[i] = 1);

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15 };
		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);

		ConnectionCalculator conv = OperationsFactory.conv2D();
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

	@Test
	public void testConvolutionsStride2()
	{
		testConvolutionsStride2(Runtime.CPU_SEQ);
		testConvolutionsStride2(Runtime.OPENCL);
	}

	private void testConvolutionsStride2(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

		Conv2DConnection c = new ConnectionFactory().conv2d(new Layer(), new Layer(), 4, 4, 1, 2, 2, 1, 2, 2, 0, 0);

		c.getWeights().forEach(i -> c.getWeights().getElements()[i] = 1);

		ValuesProvider vp = TensorFactory.tensorProvider(c, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		float[] src = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 8f, 7.5f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15 };
		System.arraycopy(src, 0, vp.get(c.getInputLayer()).getElements(), vp.get(c.getInputLayer()).getStartIndex(), src.length);

		ConnectionCalculator conv = OperationsFactory.conv2D();
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

	@Test
	public void testDimensions()
	{
		testDimensions(Runtime.CPU_SEQ);
		testDimensions(Runtime.OPENCL);
	}

	private void testDimensions(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

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

	@Test
	public void testCNNConstruction()
	{
		testCNNConstruction(Runtime.CPU_SEQ);
		testCNNConstruction(Runtime.OPENCL);
	}

	private void testCNNConstruction(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

		NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 32, 32, 1 }, { 5, 5, 6, 1, 1, 0, 0 }, { 2, 2, 2, 2, 0, 0 }, { 5, 5, 16, 1, 1, 0, 0 }, { 2, 2, 2, 2, 0, 0 }, { 5, 5, 120, 1, 1, 0, 0 },
				{ 84 }, { 10 } }, true);
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
	public void testCNNConstruction2()
	{
		testCNNConstruction2(Runtime.CPU_SEQ);
		testCNNConstruction2(Runtime.OPENCL);
	}

	private void testCNNConstruction2(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

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

	@Test
	public void testCNNConstruction3()
	{
		testCNNConstruction3(Runtime.CPU_SEQ);
		testCNNConstruction3(Runtime.OPENCL);
	}

	private void testCNNConstruction3(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

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

	@Test
	public void testCNNLayerCalculatorConstruction()
	{
		testCNNLayerCalculatorConstruction(Runtime.CPU_SEQ);
		testCNNLayerCalculatorConstruction(Runtime.OPENCL);
	}

	private void testCNNLayerCalculatorConstruction(Runtime runtime)
	{
		// prepare engine
		configureGlobalRuntimeEnvironment(runtime);

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
}
