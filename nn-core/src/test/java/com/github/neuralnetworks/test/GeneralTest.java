package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.stream.IntStream;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.input.ScalingInputFunction;
import com.github.neuralnetworks.input.SimpleFileInputProvider;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.input.image.CompositeAugmentStrategy;
import com.github.neuralnetworks.input.image.Point5CropAugmentStrategy;
import com.github.neuralnetworks.input.image.RandomImageInputProvider;
import com.github.neuralnetworks.input.image.VerticalFlipAugmentStrategy;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputDataImpl;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.Util;

@RunWith(Parameterized.class)
public class GeneralTest 
{
	public GeneralTest(RuntimeConfiguration conf)
	{
		super();
		Environment.getInstance().setRuntimeConfiguration(conf);
	}

	@Parameters
	public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
	{
		RuntimeConfiguration conf1 = new RuntimeConfiguration();
		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);

		RuntimeConfiguration conf2 = new RuntimeConfiguration();
		conf2.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		conf2.setUseDataSharedMemory(true);
		conf2.setUseWeightsSharedMemory(true);

		return Arrays.asList(new RuntimeConfiguration[][] { { conf1 }, { conf2 } });
	}

	@Test
	@Ignore
	@Deprecated // test moved
	public void testTensorProvider()
	{
		ValuesProvider tp = new ValuesProvider(true);
		String s1 = "1";
		tp.add(s1, 2, 3);
		String s2 = "2";
		tp.add(s2, 2, 3);

		assertTrue(tp.get(s1).getElements() == tp.get(s2).getElements());
		assertEquals(12, tp.get(s1).getElements().length, 0);
		assertEquals(0, tp.get(s1).getStartIndex(), 0);
		assertEquals(12, tp.get(s2).getElements().length, 0);
		assertEquals(6, tp.get(s2).getStartIndex(), 0);

		ValuesProvider tp2 = new ValuesProvider(tp);
		tp2.add(s1, 2, 3);
		tp2.add(s2, 2, 3);

		assertTrue(tp2.get(s1).getElements() == tp.get(s2).getElements());
		assertEquals(24, tp2.get(s1).getElements().length, 0);
		assertEquals(12, tp2.get(s1).getStartIndex(), 0);
		assertEquals(18, tp2.get(s2).getStartIndex(), 0);
	}

	@Test
	@Ignore
	@Deprecated // test moved
	public void testTensorProvider2()
	{
		boolean sharedMemory = Environment.getInstance().getRuntimeConfiguration().getUseWeightsSharedMemory();
		Environment.getInstance().getRuntimeConfiguration().setUseWeightsSharedMemory(true);

		NeuralNetworkImpl nn = new NeuralNetworkImpl();

		Layer i = new Layer();
		Layer h = new Layer();
		Layer o = new Layer();

		nn.addLayer(i);

		Environment.getInstance().getRuntimeConfiguration().setUseWeightsSharedMemory(true);
		ConnectionFactory cf = new ConnectionFactory();
		NNFactory.addFullyConnectedLayer(nn, h, cf, 2, 3, true);
		NNFactory.addFullyConnectedLayer(nn, o, cf, 4, 1, true);

		ValuesProvider tp = TensorFactory.tensorProvider(nn, 2, true);

		Matrix im = tp.get(nn.getInputLayer());
		Matrix hm1 = tp.get(h, 2, 3);
		Matrix hm2 = tp.get(h, 2, 4);

		Tensor om = tp.get(o);

		assertTrue(im == tp.get(i, 2, 2));
		assertTrue(im == tp.get(i));
		assertTrue(hm1 == tp.get(h, 2, 3));
		assertTrue(hm2 == tp.get(h, 2, 4));
		assertTrue(hm1 == TensorFactory.tensor(h, nn.getConnection(i, h), tp));
		assertTrue(hm2 == TensorFactory.tensor(h, nn.getConnection(h, o), tp));
		assertTrue(om == tp.get(o, 2, 1));
		assertTrue(om == tp.get(o));

		Environment.getInstance().getRuntimeConfiguration().setUseWeightsSharedMemory(sharedMemory);
	}

	@Test
	@Ignore
	@Deprecated // test moved
	public void testTensorProvider3()
	{
		// simple mlp test
		Environment.getInstance().getRuntimeConfiguration().setUseWeightsSharedMemory(true);
		NeuralNetworkImpl nn = NNFactory.mlp(new int[] { 3, 4, 2 }, true);
		ValuesProvider tp = TensorFactory.tensorProvider(nn, 2, false);

		Matrix in = tp.get(nn.getInputLayer());
		Matrix hidden = tp.get(nn.getLayers().stream().filter(l -> l != nn.getInputLayer() && l != nn.getOutputLayer() && !Util.isBias(l)).findFirst().get());
		Matrix out = tp.get(nn.getOutputLayer());

		assertEquals(6, in.getElements().length, 0);
		assertEquals(2, in.getRows(), 0);
		assertEquals(3, in.getColumns(), 0);
		assertEquals(8, hidden.getElements().length, 0);
		assertEquals(2, hidden.getRows(), 0);
		assertEquals(4, hidden.getColumns(), 0);
		assertEquals(4, out.getElements().length, 0);
		assertEquals(2, out.getRows(), 0);
		assertEquals(2, out.getColumns(), 0);

		hidden.set(2, 0, 1);
		hidden.set(3, 0, 2);
		hidden.set(4, 1, 2);
		hidden.set(5, 1, 3);
		assertEquals(2, hidden.get(0, 1), 0);
		assertEquals(3, hidden.get(0, 2), 0);
		assertEquals(4, hidden.get(1, 2), 0);
		assertEquals(5, hidden.get(1, 3), 0);

		out.set(8, 0, 1);
		out.set(9, 1, 1);
		assertEquals(8, out.get(0, 1), 0);
		assertEquals(9, out.get(1, 1), 0);
	}

	@Test
	@Ignore
	@Deprecated // test moved
	public void testConnectionFactory()
	{
		boolean sharedMemory = Environment.getInstance().getRuntimeConfiguration().getUseWeightsSharedMemory();
		Environment.getInstance().getRuntimeConfiguration().setUseWeightsSharedMemory(true);

		ConnectionFactory f = new ConnectionFactory();
		FullyConnected fc1 = f.fullyConnected(null, null, 2, 3);
		FullyConnected fc2 = f.fullyConnected(null, null, 5, 2);

		assertTrue(fc1.getWeights().getElements() == fc2.getWeights().getElements());
		assertEquals(16, fc1.getWeights().getElements().length, 0);
		assertEquals(0, fc1.getWeights().getStartOffset(), 0);
		assertEquals(3, fc1.getWeights().getRows(), 0);
		assertEquals(2, fc1.getWeights().getColumns(), 0);
		fc1.getWeights().set(3, 1, 1);
		assertEquals(3, fc1.getWeights().get(1, 1), 0);

		assertEquals(6, fc2.getWeights().getStartOffset(), 0);
		assertEquals(2, fc2.getWeights().getRows(), 0);
		assertEquals(5, fc2.getWeights().getColumns(), 0);
		fc2.getWeights().set(5, 1, 1);
		assertEquals(5, fc2.getWeights().get(1, 1), 0);

		Conv2DConnection c = f.conv2d(null, null, 3, 3, 3, 2, 2, 3, 1, 1, 0, 0);
		assertEquals(52, c.getWeights().getElements().length, 0);
		assertEquals(36, c.getWeights().getSize(), 0);
		assertEquals(16, c.getWeights().getStartOffset(), 0);
		assertEquals(4, c.getWeights().getDimensions().length, 0);

		Environment.getInstance().getRuntimeConfiguration().setUseWeightsSharedMemory(sharedMemory);
	}

	@Test
	@Ignore
	@Deprecated // test moved
	public void testRandomInitializer()
	{
		NeuralNetworkImpl nn = NNFactory.mlp(new int[] { 3, 2 }, true);
		NNRandomInitializer rand = new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.1f, 0.1f), 0.5f);
		rand.initialize(nn);

		for (Layer l : nn.getLayers())
		{
			if (Util.isBias(l))
			{
				Tensor t = ((FullyConnected) l.getConnections().get(0)).getWeights();
				float[] elements = t.getElements();
				t.forEach(i -> assertEquals(0.5, elements[i], 0f));
			} else
			{
				Tensor t = ((FullyConnected) l.getConnections().get(0)).getWeights();
				float[] elements = t.getElements();
				t.forEach(i -> assertTrue(elements[i] >= -0.1f && elements[i] <= 0.1f && elements[i] != 0));
			}
		}
	}

	@Test
	@Ignore
	@Deprecated // test moved
	public void testRandomInitializer1()
	{
		NeuralNetworkImpl nn = NNFactory.mlp(new int[] { 3, 2 }, true);

		NNRandomInitializer rand = new NNRandomInitializer(new MersenneTwisterRandomInitializer(2f, 3f), new MersenneTwisterRandomInitializer(-2f, -1f));
		rand.initialize(nn);

		for (Layer l : nn.getLayers())
		{
			if (Util.isBias(l))
			{
				Tensor t = ((FullyConnected) l.getConnections().get(0)).getWeights();
				float[] elements = t.getElements();
				t.forEach(i -> assertTrue(elements[i] >= -2f && elements[i] <= -1f));
			} else
			{
				Tensor t = ((FullyConnected) l.getConnections().get(0)).getWeights();
				float[] elements = t.getElements();
				t.forEach(i -> assertTrue(elements[i] >= 2f && elements[i] <= 3f));
			}
		}
	}

	@Test
	@Ignore
	@Deprecated
	public void testTensor()
	{
		Tensor t = TensorFactory.tensor(2, 2, 2);
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

		Iterator<Integer> it = t.iterator();
		for (int i = 0; i < elements.length && it.hasNext(); i++)
		{
			assertEquals(i + 1, elements[i], 0);
			assertEquals(i + 1, elements[it.next()], 0);
		}

		t = TensorFactory.tensor(5, 5, 5);
		assertEquals(25, t.getDimensionElementsDistance(0), 0);
		assertEquals(5, t.getDimensionElementsDistance(1), 0);
		assertEquals(1, t.getDimensionElementsDistance(2), 0);

		elements = t.getElements();

		for (int i = 0; i < elements.length; i++)
		{
			elements[i] = i + 1;
		}

		Tensor t2 = TensorFactory.tensor(t, new int[][] { { 3, 0, 0 }, { 4, 4, 4 } }, true);
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

		Tensor[] tarr = TensorFactory.tensor(new int[] { 2, 2, 2 }, new int[] { 3, 3 });
		assertEquals(17, tarr[0].getElements().length, 0);
		assertEquals(0, tarr[0].getStartOffset(), 0);
		assertEquals(8, tarr[1].getStartOffset(), 0);
		assertTrue(tarr[1] instanceof Matrix);

		IntStream.range(0, tarr[0].getElements().length).forEach(i -> tarr[0].getElements()[i] = i + 1);
		assertEquals(7, tarr[0].get(1, 1, 0), 0);
		assertEquals(13, tarr[1].get(1, 1), 0);
	}

	@Test
	@Ignore
	@Deprecated
	public void testMatrix()
	{
		Matrix m = TensorFactory.tensor(5, 6);

		assertEquals(5, m.getRows(), 0);
		assertEquals(6, m.getColumns(), 0);
		assertEquals(6, m.getDimensionElementsDistance(0), 0);
		assertEquals(1, m.getDimensionElementsDistance(1), 0);
		assertEquals(5, m.getDimensions()[0], 0);
		assertEquals(6, m.getDimensions()[1], 0);

		for (int i = 0; i < m.getElements().length; i++)
		{
			m.getElements()[i] = i + 1;
		}

		assertEquals(2, m.get(0, 1), 0);
		assertEquals(15, m.get(2, 2), 0);

		m = TensorFactory.tensor(1, 6);
		for (int i = 0; i < m.getElements().length; i++)
		{
			m.getElements()[i] = i + 1;
		}

		assertEquals(2, m.get(0, 1), 0);
		assertEquals(6, m.get(0, 5), 0);

		m = TensorFactory.tensor(6, 1);
		for (int i = 0; i < m.getElements().length; i++)
		{
			m.getElements()[i] = i + 1;
		}

		assertEquals(2, m.get(1, 0), 0);
		assertEquals(6, m.get(5, 0), 0);

		// submatrix
		Tensor t = TensorFactory.tensor(5, 5, 5);
		float[] elements = t.getElements();

		for (int i = 0; i < elements.length; i++)
		{
			elements[i] = i + 1;
		}

		m = TensorFactory.tensor(t, new int[][] { { 1, 0, 0 }, { 1, 4, 4 } }, true);
		assertEquals(26, m.get(0, 0), 0);
		assertEquals(27, m.get(0, 1), 0);
		assertEquals(36, m.get(2, 0), 0);
		assertEquals(38, m.get(2, 2), 0);

		m = TensorFactory.tensor(t, new int[][] { { 1, 0, 0 }, { 1, 4, 4 } }, true);
		assertEquals(26, m.get(0, 0), 0);
		assertEquals(27, m.get(0, 1), 0);
		assertEquals(36, m.get(2, 0), 0);
		assertEquals(38, m.get(2, 2), 0);

		m = TensorFactory.tensor(t, new int[][] { { 0, 0, 1 }, { 4, 4, 1 } }, true);
		assertEquals(2, m.get(0, 0), 0);
		assertEquals(7, m.get(0, 1), 0);
		assertEquals(12, m.get(0, 2), 0);
		assertEquals(27, m.get(1, 0), 0);
		assertEquals(32, m.get(1, 1), 0);
		assertEquals(37, m.get(1, 2), 0);

		m = TensorFactory.tensor(t, new int[][] { { 2, 2, 1 }, { 3, 3, 1 } }, true);
		assertEquals(62, m.get(0, 0), 0);
		assertEquals(67, m.get(0, 1), 0);
		assertEquals(92, m.get(1, 1), 0);
		Iterator<Integer> it = m.iterator();

		assertEquals(62, m.getElements()[it.next()], 0);
		assertEquals(67, m.getElements()[it.next()], 0);
		it.next();
		assertEquals(92, m.getElements()[it.next()], 0);

		it = m.iterator(new int[][] { { 1, 0 }, { 1, 1 } });
		it.next();
		assertEquals(92, m.getElements()[it.next()], 0);

		m = TensorFactory.tensor(4, 4);
		for (int i = 0; i < m.getElements().length; i++)
		{
			m.getElements()[i] = i + 1;
		}

		Matrix m2 = TensorFactory.tensor(m, new int[][] { { 1, 1 }, { 2, 2 } }, true);
		assertEquals(6, m2.get(0, 0), 0);
		assertEquals(7, m2.get(0, 1), 0);
		assertEquals(10, m2.get(1, 0), 0);
		assertEquals(11, m2.get(1, 1), 0);
	}

	@Test
	@Ignore
	@Deprecated
	public void testScaling()
	{
		float[][] input = new float[][] { { 1, 3 }, { -1.5f, 1.5f } };
		ScalingInputFunction si = new ScalingInputFunction();
		Matrix m = TensorFactory.matrix(input);
		si.value(m);

		assertEquals(0f, m.get(0, 0), 0);
		assertEquals(1f, m.get(0, 1), 0);
		assertEquals(0f, m.get(1, 0), 0);
		assertEquals(1f, m.get(1, 1), 0);
	}

	@Test
	@Ignore
	@Deprecated
	public void testImageInputProvider()
	{
		RandomImageInputProvider ri = new RandomImageInputProvider(16, 256, 256, 150528, 100);
		ri.getProperties().setAugmentedImagesBufferSize(1024);
		ri.getProperties().setImagesBulkSize(32);
		ri.getProperties().setGroupByChannel(true);
		ri.getProperties().setSubtractMean(true);
		ri.getProperties().setParallelPreprocessing(true);
		ri.getProperties().setScaleColors(true);

		CompositeAugmentStrategy cas = new CompositeAugmentStrategy();
		cas.addStrategy(new VerticalFlipAugmentStrategy());
		cas.addStrategy(new Point5CropAugmentStrategy(224, 224));
		ri.getProperties().setAugmentStrategy(cas);

		Tensor t = TensorFactory.tensor(32, 224, 224, 3);

		ri.getNextInput(t);
		ri.getNextInput(t);

		assertEquals(0, ri.getAugmentedToRaw().size() % 320, 0);

		if (ri.getProperties().getSubtractMean())
		{
			t.forEach(i -> assertTrue(t.getElements()[i] >= -1 && t.getElements()[i] <= 1));
		} else
		{
			t.forEach(i -> assertTrue(t.getElements()[i] >= 0 && t.getElements()[i] <= 1));
		}

		for (float[] arr : ri.getValues())
		{
			assertEquals(150528, arr.length);

			for (float f : arr)
			{
				if (ri.getProperties().getSubtractMean())
				{
					assertTrue(f >= -1 && f <= 1);
				} else
				{
					assertTrue(f >= 0 && f <= 1);
				}
			}
		}
	}

	@Test
	@Ignore
	@Deprecated
	public void testSimpleInputProvider() {
		SimpleInputProvider sip = new SimpleInputProvider(new float[][] { {1, 1, 1}, {0, 0, 0} }, new float[][] { {1, 1}, {0, 0} });
		TrainingInputData ti = new TrainingInputDataImpl(TensorFactory.tensor(2, 3), TensorFactory.tensor(2, 2));
		sip.populateNext(ti);

		assertEquals(1, ti.getInput().get(0, 0), 0);
		assertEquals(1, ti.getInput().get(0, 1), 0);
		assertEquals(1, ti.getInput().get(0, 2), 0);
		assertEquals(0, ti.getInput().get(1, 0), 0);
		assertEquals(0, ti.getInput().get(1, 1), 0);
		assertEquals(0, ti.getInput().get(1, 2), 0);

		assertEquals(1, ti.getTarget().get(0, 0), 0);
		assertEquals(1, ti.getTarget().get(0, 1), 0);
		assertEquals(0, ti.getTarget().get(1, 0), 0);
		assertEquals(0, ti.getTarget().get(1, 1), 0);
	}


	@Test
	@Ignore
	@Deprecated
	public void testHyperparameters() {
		Hyperparameters hp = new Hyperparameters();
		hp.setDefaultLearningRate(0.1f);

		Object o = new Object();
		hp.setLearningRate(o, 0.5f);

		assertEquals(0.1f, hp.getDefaultLearningRate(), 0);
		assertEquals(0.5f, hp.getLearningRate(o), 0);
		assertEquals(0.1f, hp.getLearningRate(new Object()), 0);
	}

	@Test
	@Ignore
	@Deprecated
	public void testSimpleFileInputProvider() {
		SimpleFileInputProvider sip2 = null;
		try
		{
			SimpleInputProvider sip1 = new SimpleInputProvider(new float[][] { { 1, 2 }, { 3, 4 }, { 5, 6 } }, new float[][] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
			Util.inputToFloat(sip1, "ip_input", "ip_target");
			sip1.reset();

			sip2 = new SimpleFileInputProvider("ip_input", "ip_target", sip1.getInputDimensions(), sip1.getTargetDimensions(), sip1.getInputSize());
			TrainingInputDataImpl ti1 = new TrainingInputDataImpl(TensorFactory.tensor(2, sip1.getInputDimensions()), TensorFactory.tensor(2, sip1.getTargetDimensions()));
			TrainingInputDataImpl ti2 = new TrainingInputDataImpl(TensorFactory.tensor(2, sip2.getInputDimensions()), TensorFactory.tensor(2, sip2.getTargetDimensions()));

			for (int i = 0; i < sip2.getInputSize(); i += 2)
			{
				sip1.populateNext(ti1);
				sip2.populateNext(ti2);
				assertTrue(Arrays.equals(ti1.getInput().getElements(), ti2.getInput().getElements()));
				assertTrue(Arrays.equals(ti1.getTarget().getElements(), ti2.getTarget().getElements()));
			}
		} finally
		{
			try
			{
				sip2.close();
				Files.delete(Paths.get("ip_input"));
				Files.delete(Paths.get("ip_target"));
			} catch (IOException e)
			{
				e.printStackTrace();
			}
		}
	}
}
