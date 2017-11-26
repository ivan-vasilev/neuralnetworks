package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
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
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.BreadthFirstOrderStrategy;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.LayerOrderStrategy.ConnectionCandidate;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorTensorFunctions;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.calculation.operations.cpu.MaxoutWinners;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.tensor.Tensor.TensorIterator;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationLayerCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.backpropagation.LossFunction;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

/**
 * General feedforward neural networks tests
 */
@RunWith(Parameterized.class)
public class FFNNTest
{
	public FFNNTest(RuntimeConfiguration conf)
	{
		Environment.getInstance().setRuntimeConfiguration(conf);
	}

	@Parameters
	public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> configurations = new ArrayList<>();

//		RuntimeConfiguration conf1 = new RuntimeConfiguration();
//		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//		conf1.setUseDataSharedMemory(false);
//		conf1.setUseWeightsSharedMemory(false);
//		configurations.add(new RuntimeConfiguration[] { conf1 });
//
//		RuntimeConfiguration conf2 = new RuntimeConfiguration();
//		conf2.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//		conf2.setUseDataSharedMemory(true);
//		conf2.setUseWeightsSharedMemory(true);
//		configurations.add(new RuntimeConfiguration[] { conf2 });

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
	@Deprecated
	public void testWeightedSumFFSimple() {
		Layer il1 = new Layer();
		Layer ol = new Layer();

		Tensor weights = TensorFactory.tensor(2, 2, 3);

		FullyConnected c1 = new FullyConnected(il1, ol, TensorFactory.tensor(weights, new int[][] { { 0, 0, 0 }, { 0, 2, 3 } }, true));

		Matrix cg = c1.getWeights();
		cg.set(1, 0, 0);
		cg.set(2, 0, 1);
		cg.set(3, 0, 2);
		cg.set(4, 1, 0);
		cg.set(5, 1, 1);
		cg.set(6, 1, 2);

		List<Connections> connections = new ArrayList<>();
		connections.add(c1);

		NeuralNetworkImpl nn = new NeuralNetworkImpl();
		nn.addConnections(connections.toArray(new Connections[connections.size()]));

		ValuesProvider vp = TensorFactory.tensorProvider(nn, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

		Matrix i1 = vp.get(nn.getInputLayer());
		i1.set(1, 0, 0);
		i1.set(2, 0, 1);
		i1.set(3, 0, 2);
		i1.set(4, 1, 0);
		i1.set(5, 1, 1);
		i1.set(6, 1, 2);

		ConnectionCalculator aws = OperationsFactory.weightedSum();
		aws.calculate(connections, vp, ol);

		// most simple case
		Matrix o = vp.get(nn.getOutputLayer());
		assertEquals(14, o.get(0, 0), 0);
		assertEquals(32, o.get(1, 0), 0);
		assertEquals(32, o.get(0, 1), 0);
		assertEquals(77, o.get(1, 1), 0);
	}

	@Ignore
	@Test
	@Deprecated /// test moved to OpenCLWeightedSumTest
	public void testWeightedSumFF()
	{
		Layer il1 = new Layer();
		Layer ol = new Layer();
		Layer il2 = new Layer();

		Tensor weights = TensorFactory.tensor(2, 2, 3);

		FullyConnected c1 = new FullyConnected(il1, ol, TensorFactory.tensor(weights, new int[][] { { 0, 0, 0 }, { 0, 1, 2 } }, true));
		FullyConnected c2 = new FullyConnected(il2, ol, TensorFactory.tensor(weights, new int[][] { { 1, 0, 0 }, { 1, 1, 2 } }, true));
		FullyConnected bc = new FullyConnected(new Layer(), ol, 1, 2);

		Matrix cg = c1.getWeights();
		cg.set(1, 0, 0);
		cg.set(2, 0, 1);
		cg.set(3, 0, 2);
		cg.set(4, 1, 0);
		cg.set(5, 1, 1);
		cg.set(6, 1, 2);

		cg = c2.getWeights();
		cg.set(1, 0, 0);
		cg.set(2, 0, 1);
		cg.set(3, 0, 2);
		cg.set(4, 1, 0);
		cg.set(5, 1, 1);
		cg.set(6, 1, 2);

		Matrix bcg = bc.getWeights();
		bcg.set(0.1f, 0, 0);
		bcg.set(0.2f, 1, 0);

		List<Connections> connections = new ArrayList<>();
		connections.add(c1);

		NeuralNetworkImpl nn = new NeuralNetworkImpl();
		nn.addConnections(connections.toArray(new Connections[connections.size()]));

		ValuesProvider vp = TensorFactory.tensorProvider(nn, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

		Matrix i1 = vp.get(nn.getInputLayer());
		i1.set(1, 0, 0);
		i1.set(2, 0, 1);
		i1.set(3, 0, 2);
		i1.set(4, 1, 0);
		i1.set(5, 1, 1);
		i1.set(6, 1, 2);

		ConnectionCalculator aws = OperationsFactory.weightedSum();
		aws.calculate(connections, vp, ol);

		// most simple case
		Matrix o = vp.get(nn.getOutputLayer());
		assertEquals(14, o.get(0, 0), 0);
		assertEquals(32, o.get(0, 1), 0);
		assertEquals(32, o.get(1, 0), 0);
		assertEquals(77, o.get(1, 1), 0);

		// with bias
		connections = new ArrayList<>();
		connections.add(c1);
		connections.add(bc);

		nn = new NeuralNetworkImpl();
		nn.addConnections(connections.toArray(new Connections[connections.size()]));
		vp = TensorFactory.tensorProvider(nn, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

		i1 = vp.get(nn.getInputLayer());
		i1.set(1, 0, 0);
		i1.set(2, 0, 1);
		i1.set(3, 0, 2);
		i1.set(4, 1, 0);
		i1.set(5, 1, 1);
		i1.set(6, 1, 2);

		Tensor b1 = vp.get(bc.getInputLayer());
		b1.set(1, 0, 0);
		b1.set(1, 1, 0);

		ConnectionCalculator aws2 = OperationsFactory.weightedSum();
		aws2.calculate(connections, vp, ol);

		o = vp.get(nn.getOutputLayer());
		assertEquals(14.1, o.get(0, 0), 0.01);
		assertEquals(32.1, o.get(1, 0), 0.01);
		assertEquals(32.2, o.get(0, 1), 0.01);
		assertEquals(77.2, o.get(1, 1), 0.01);

		// combined layers
		connections = new ArrayList<>();
		connections.add(c1);
		connections.add(c2);
		connections.add(bc);
		nn = new NeuralNetworkImpl();
		nn.addConnections(connections.toArray(new Connections[connections.size()]));
		vp = TensorFactory.tensorProvider(nn, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

		i1 = vp.get(il1);
		i1.set(1, 0, 0);
		i1.set(2, 0, 1);
		i1.set(3, 0, 2);
		i1.set(4, 1, 0);
		i1.set(5, 1, 1);
		i1.set(6, 1, 2);

		Matrix i2 = vp.get(il2);
		i2.set(1, 0, 0);
		i2.set(2, 0, 1);
		i2.set(3, 0, 2);
		i2.set(4, 1, 0);
		i2.set(5, 1, 1);
		i2.set(6, 1, 2);

		b1 = vp.get(bc.getInputLayer());
		b1.set(1, 0, 0);
		b1.set(1, 1, 0);

		aws2 = OperationsFactory.weightedSum();
		aws2.calculate(connections, vp, ol);

		o = vp.get(nn.getOutputLayer());
		assertEquals(28.1, o.get(0, 0), 0.01);
		assertEquals(64.1, o.get(1, 0), 0.01);
		assertEquals(64.2, o.get(0, 1), 0.01);
		assertEquals(154.2, o.get(1, 1), 0.01);
	}

	@Ignore
	@Test
	@Deprecated /// test moved to OpenCLWeightedSumTest
	public void testWeightedSumBP()
	{
		Layer il1 = new Layer();
		Layer ol = new Layer();
		Layer il2 = new Layer();

		Tensor weights = TensorFactory.tensor(2, 3, 2);
		FullyConnected c1 = new FullyConnected(ol, il1, TensorFactory.tensor(weights, new int[][] { { 0, 0, 0 }, { 0, 2, 1 } }, true));
		FullyConnected c2 = new FullyConnected(ol, il2, TensorFactory.tensor(weights, new int[][] { { 1, 0, 0 }, { 1, 2, 1 } }, true));
		FullyConnected bc = new FullyConnected(new Layer(), ol, 1, 2);

		Matrix cg = c1.getWeights();
		cg.set(1, 0, 0);
		cg.set(2, 1, 0);
		cg.set(3, 2, 0);
		cg.set(4, 0, 1);
		cg.set(5, 1, 1);
		cg.set(6, 2, 1);

		cg = c2.getWeights();
		cg.set(1, 0, 0);
		cg.set(2, 1, 0);
		cg.set(3, 2, 0);
		cg.set(4, 0, 1);
		cg.set(5, 1, 1);
		cg.set(6, 2, 1);

		Matrix bcg = bc.getWeights();
		bcg.set(0.1f, 0, 0);
		bcg.set(0.2f, 1, 0);

		ConnectionCalculator aws = OperationsFactory.weightedSum();

		List<Connections> connections = new ArrayList<>();
		connections.add(c1);
		NeuralNetworkImpl nn = new NeuralNetworkImpl();
		nn.addConnections(connections.toArray(new Connections[connections.size()]));
		ValuesProvider vp = TensorFactory.tensorProvider(nn, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

		Matrix i1 = vp.get(il1);
		i1.set(1, 0, 0);
		i1.set(2, 0, 1);
		i1.set(3, 0, 2);
		i1.set(4, 1, 0);
		i1.set(5, 1, 1);
		i1.set(6, 1, 2);

		aws.calculate(connections, vp, ol);

		// most simple case
		Matrix o = vp.get(ol);
		assertEquals(14, o.get(0, 0), 0);
		assertEquals(32, o.get(1, 0), 0);
		assertEquals(32, o.get(0, 1), 0);
		assertEquals(77, o.get(1, 1), 0);

		// with bias
		connections = new ArrayList<>();
		connections.add(c1);
		connections.add(bc);
		nn = new NeuralNetworkImpl();
		nn.addConnections(connections.toArray(new Connections[connections.size()]));
		vp = TensorFactory.tensorProvider(nn, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		i1 = vp.get(il1);
		i1.set(1, 0, 0);
		i1.set(2, 0, 1);
		i1.set(3, 0, 2);
		i1.set(4, 1, 0);
		i1.set(5, 1, 1);
		i1.set(6, 1, 2);

		Tensor b1 = vp.get(bc.getInputLayer());
		b1.set(1, 0, 0);
		b1.set(1, 1, 0);

		ConnectionCalculator aws2 = OperationsFactory.weightedSum();
		aws2.calculate(connections, vp, ol);

		o = vp.get(ol);
		assertEquals(14.1, o.get(0, 0), 0.01);
		assertEquals(32.1, o.get(1, 0), 0.01);
		assertEquals(32.2, o.get(0, 1), 0.01);
		assertEquals(77.2, o.get(1, 1), 0.01);

		// combined layers
		connections = new ArrayList<>();
		connections.add(c1);
		connections.add(c2);
		connections.add(bc);
		nn = new NeuralNetworkImpl();
		nn.addConnections(connections.toArray(new Connections[connections.size()]));
		vp = TensorFactory.tensorProvider(nn, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

		i1 = vp.get(il1);
		i1.set(1, 0, 0);
		i1.set(2, 0, 1);
		i1.set(3, 0, 2);
		i1.set(4, 1, 0);
		i1.set(5, 1, 1);
		i1.set(6, 1, 2);

		Matrix i2 = vp.get(il2);
		i2.set(1, 0, 0);
		i2.set(2, 0, 1);
		i2.set(3, 0, 2);
		i2.set(4, 1, 0);
		i2.set(5, 1, 1);
		i2.set(6, 1, 2);

		b1 = vp.get(bc.getInputLayer());
		b1.set(1, 0, 0);
		b1.set(1, 1, 0);

		aws2 = OperationsFactory.weightedSum();
		aws2.calculate(connections, vp, ol);

		o = vp.get(ol);
		assertEquals(28.1, o.get(0, 0), 0.01);
		assertEquals(64.1, o.get(1, 0), 0.01);
		assertEquals(64.2, o.get(0, 1), 0.01);
		assertEquals(154.2, o.get(1, 1), 0.01);
	}

	/**
	 * Simple backpropagation test with specific values
	 */
	@Test
	@Ignore
	@Deprecated
	public void testSigmoidBP()
	{
		NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 2, 2, 1 }, false);

		FullyConnected c1 = (FullyConnected) mlp.getInputLayer().getConnections().iterator().next();
		Matrix cg1 = c1.getWeights();
		cg1.set(0.1f, 0, 0);
		cg1.set(0.8f, 0, 1);
		cg1.set(0.4f, 1, 0);
		cg1.set(0.6f, 1, 1);

		FullyConnected c2 = (FullyConnected) mlp.getOutputLayer().getConnections().iterator().next();
		Matrix cg2 = c2.getWeights();
		cg2.set(0.3f, 0, 0);
		cg2.set(0.9f, 0, 1);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }), new SimpleInputProvider(
				new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }), new MultipleNeuronsOutputError(), null, 1f, 0f, 0f, 0f, 0f, 1, 1, 1);

		bpt.train();
		bpt.test();

		assertEquals(0.272392, cg2.get(0, 0), 0.001);
		assertEquals(0.87305, cg2.get(0, 1), 0.001);
		assertEquals(0.09916, cg1.get(0, 0), 0.01);
		assertEquals(0.7978, cg1.get(0, 1), 0.01);
		assertEquals(0.3972, cg1.get(1, 0), 0.01);
		assertEquals(0.5928, cg1.get(1, 1), 0.01);
	}

	/**
	 * Simple backpropagation test with specific values
	 * https://blog.itu.dk/MAIG-E2013/files/2013/09/9point2-classification-by-backpropagation.pdf
	 */
	@Test
	@Ignore
	@Deprecated
	public void testSigmoidBP2()
	{
		NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 3, 2, 1 }, true);

		List<Connections> c = mlp.getConnections();
		FullyConnected c1 = (FullyConnected) c.get(0);
		Matrix cg1 = c1.getWeights();
		cg1.set(0.2f, 0, 0);
		cg1.set(0.4f, 0, 1);
		cg1.set(-0.5f, 0, 2);
		cg1.set(-0.3f, 1, 0);
		cg1.set(0.1f, 1, 1);
		cg1.set(0.2f, 1, 2);

		FullyConnected cb1 = (FullyConnected) c.get(1);
		Matrix cgb1 = cb1.getWeights();
		cgb1.set(-0.4f, 0, 0);
		cgb1.set(0.2f, 1, 0);

		FullyConnected c2 = (FullyConnected) c.get(2);
		Matrix cg2 = c2.getWeights();
		cg2.set(-0.3f, 0, 0);
		cg2.set(-0.2f, 0, 1);

		FullyConnected cb2 = (FullyConnected) c.get(3);
		Matrix cgb2 = cb2.getWeights();
		cgb2.set(0.1f, 0, 0);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 1, 0, 1 } }, new float[][] { { 1 } }), new SimpleInputProvider(new float[][] { { 1,
				0, 1 } }, new float[][] { { 1 } }), null, null, 0.9f, 0f, 0f, 0f, 0f, 1, 1, 1);
		bpt.train();

		assertEquals(-0.261, cg2.get(0, 0), 0.001);
		assertEquals(-0.138, cg2.get(0, 1), 0.001);

		assertEquals(0.218, cgb2.get(0, 0), 0.001);

		assertEquals(0.192, cg1.get(0, 0), 0.001);
		assertEquals(0.4, cg1.get(0, 1), 0.001);
		assertEquals(-0.508, cg1.get(0, 2), 0.001);
		assertEquals(-0.306, cg1.get(1, 0), 0.001);
		assertEquals(0.1, cg1.get(1, 1), 0.001);
		assertEquals(0.194, cg1.get(1, 2), 0.001);

		assertEquals(-0.408, cgb1.get(0, 0), 0.001);
		assertEquals(0.194, cgb1.get(1, 0), 0.001);
	}

	@Test
	@Ignore
	@Deprecated
	public void testSigmoidBP3()
	{
		NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 3, 2, 1 }, true);

		List<Connections> c = mlp.getConnections();
		FullyConnected c1 = (FullyConnected) c.get(0);
		Matrix cg1 = c1.getWeights();
		cg1.set(0.2f, 0, 0);
		cg1.set(0.4f, 0, 1);
		cg1.set(-0.5f, 0, 2);
		cg1.set(-0.3f, 1, 0);
		cg1.set(0.1f, 1, 1);
		cg1.set(0.2f, 1, 2);

		FullyConnected cb1 = (FullyConnected) c.get(1);
		Matrix cgb1 = cb1.getWeights();
		cgb1.set(-0.4f, 0, 0);
		cgb1.set(0.2f, 1, 0);

		FullyConnected c2 = (FullyConnected) c.get(2);
		Matrix cg2 = c2.getWeights();
		cg2.set(-0.3f, 0, 0);
		cg2.set(-0.2f, 0, 1);

		FullyConnected cb2 = (FullyConnected) c.get(3);
		Matrix cgb2 = cb2.getWeights();
		cgb2.set(0.1f, 0, 0);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 1, 0, 1 }, { 1, 1, 0 } }, new float[][] { { 1 }, { 1 } }), null, null, null, 0.9f,
				0f, 0f, 0f, 0f, 1, 1, 1);
		bpt.train();

		assertEquals(-0.1996, cg2.get(0, 0), 0.001);
		assertEquals(-0.0823, cg2.get(0, 1), 0.001);

		assertEquals(0.3302, cgb2.get(0, 0), 0.001);

		assertEquals(0.1849, cg1.get(0, 0), 0.001);
		assertEquals(0.3927, cg1.get(0, 1), 0.001);
		assertEquals(-0.508, cg1.get(0, 2), 0.001);
		assertEquals(-0.3098, cg1.get(1, 0), 0.001);
		assertEquals(0.0961, cg1.get(1, 1), 0.001);
		assertEquals(0.194, cg1.get(1, 2), 0.001);

		assertEquals(-0.4151, cgb1.get(0, 0), 0.001);
		assertEquals(0.1902, cgb1.get(1, 0), 0.001);
	}

	/**
	 * maxout ff
	 */
	@Ignore
	@Test
	@Deprecated
	public void testMaxoutFF()
	{
		NeuralNetworkImpl nn = CalculationFactory.maxout(new int[] { 2, 2 }, true);

		List<Connections> c = nn.getConnections();
		FullyConnected c1 = (FullyConnected) c.get(0);
		Matrix cg1 = c1.getWeights();
		cg1.set(0.1f, 0, 0);
		cg1.set(0.5f, 0, 1);
		cg1.set(0.1f, 1, 0);
		cg1.set(0.5f, 1, 1);

		FullyConnected cb1 = (FullyConnected) c.get(1);
		Matrix cgb1 = cb1.getWeights();
		cgb1.set(0.1f, 0, 0);
		cgb1.set(0.2f, 1, 0);

		ValuesProvider results = TensorFactory.tensorProvider(nn, 2, true);
		Matrix in = results.get(nn.getInputLayer());
		in.set(8, 0, 0);
		in.set(2, 1, 0);
		in.set(1, 0, 1);
		in.set(7, 1, 1);

		Set<Layer> calculated = new HashSet<>();
		calculated.add(nn.getInputLayer());
		nn.getLayerCalculator().calculate(nn, nn.getOutputLayer(), calculated, results);

		Matrix out = results.get(nn.getOutputLayer());
		assertEquals(1.1f, out.get(0, 0), 0f);
		assertEquals(1.2f, out.get(1, 0), 0f);
		assertEquals(3.6f, out.get(0, 1), 0f);
		assertEquals(3.7f, out.get(1, 1), 0f);

		int[] winners = MaxoutWinners.getInstance().getWinners();
		int startIndex = MaxoutWinners.getInstance().getStartPositions(c.get(0));
		assertEquals(1, winners[startIndex], 0);
		assertEquals(1, winners[startIndex + 1], 0);
	}

	/**
	 * maxout backpropagation
	 */
	@Ignore
	@Test
	@Deprecated
	public void testMaxoutBP()
	{
		NeuralNetworkImpl nn = CalculationFactory.maxout(new int[] { 2, 2 }, true);

		List<Connections> c = nn.getConnections();
		FullyConnected c1 = (FullyConnected) c.get(0);
		Matrix cg1 = c1.getWeights();
		cg1.set(0.1f, 0, 0);
		cg1.set(0.5f, 0, 1);
		cg1.set(0.1f, 1, 0);
		cg1.set(0.5f, 1, 1);

		FullyConnected cb1 = (FullyConnected) c.get(1);
		Matrix cgb1 = cb1.getWeights();
		cgb1.set(0.1f, 0, 0);
		cgb1.set(0.2f, 1, 0);

		ValuesProvider results = TensorFactory.tensorProvider(nn, 2, true);
		Matrix in = results.get(nn.getInputLayer());
		in.set(8, 0, 0);
		in.set(2, 1, 0);
		in.set(1, 0, 1);
		in.set(7, 1, 1);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, new SimpleInputProvider(new float[][] { { 8, 2 } }, new float[][] { { 1 } }), null, null, null, 0.9f, 0f, 0f, 0f, 0f, 1, 1, 1);
		bpt.train();

		assertEquals(0.1f, cg1.get(0, 0), 0f);
		assertEquals(0.5198f, cg1.get(0, 1), 0f);
		assertEquals(0.1f, cg1.get(1, 0), 0f);
		assertEquals(1.0184002f, cg1.get(1, 1), 0f);

		assertEquals(0.109900005f, cgb1.get(0, 0), 0f);
		assertEquals(0.45920008f, cgb1.get(1, 0), 0f);
	}

	@Test
	@Ignore
	@Deprecated
	public void testParallelNetworks()
	{
		ConnectionFactory cf = new ConnectionFactory();
		NeuralNetworkImpl mlp = new NeuralNetworkImpl();
		Layer input = new Layer();
		Layer leaf1 = new Layer();
		Layer leaf2 = new Layer();
		Layer output = new Layer();

		mlp.addLayer(input);

		FullyConnected fc1 = cf.fullyConnected(input, leaf1, 2, 3);
		fc1.getWeights().forEach(i -> fc1.getWeights().getElements()[i] = 0.1f);
		mlp.addConnections(fc1);

		FullyConnected fc2 = cf.fullyConnected(input, leaf2, 2, 3);
		fc2.getWeights().forEach(i -> fc2.getWeights().getElements()[i] = 0.2f);
		mlp.addConnections(fc2);

		FullyConnected fc3 = cf.fullyConnected(leaf1, output, 3, 1);
		fc3.getWeights().forEach(i -> fc3.getWeights().getElements()[i] = 0.3f);
		mlp.addConnections(fc3);
		FullyConnected fc4 = cf.fullyConnected(leaf2, output, 3, 1);
		fc4.getWeights().forEach(i -> fc4.getWeights().getElements()[i] = 0.4f);
		mlp.addConnections(fc4);

		mlp.setLayerCalculator(CalculationFactory.lcWeightedSum(mlp, null));

		Set<Layer> calculated = new HashSet<>();
		calculated.add(mlp.getInputLayer());

		ValuesProvider results = TensorFactory.tensorProvider(mlp, 1, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		results.get(mlp.getInputLayer()).set(2, 0, 0);
		results.get(mlp.getInputLayer()).set(2, 0, 1);

		mlp.getLayerCalculator().calculate(mlp, output, calculated, results);

		assertEquals(1.32, results.get(output).get(0, 0), 0.000001);
	}

	@Test
	@Ignore
	@Deprecated
	public void testRemoveLayer()
	{
		NeuralNetworkImpl mlp = NNFactory.mlp(new int[] { 3, 4, 5 }, true);
		assertEquals(5, mlp.getLayers().size(), 0);
		Layer currentOutput = mlp.getOutputLayer();
		mlp.removeLayer(mlp.getOutputLayer());
		assertEquals(3, mlp.getLayers().size(), 0);
		assertEquals(true, currentOutput != mlp.getOutputLayer());
	}

	@Test
	@Ignore
	@Deprecated
	public void testLayerOrderStrategy()
	{
		// MLP
		NeuralNetworkImpl mlp = NNFactory.mlp(new int[] { 3, 4, 5 }, true);

		List<ConnectionCandidate> ccc = new BreadthFirstOrderStrategy(mlp, mlp.getOutputLayer()).order();
		assertEquals(4, ccc.size(), 0);
		Layer l = mlp.getOutputLayer();
		assertTrue(ccc.get(0).connection == l.getConnections().get(0));
		assertTrue(ccc.get(1).connection == l.getConnections().get(1));

		l = l.getConnections().get(0).getInputLayer();
		assertTrue(ccc.get(2).connection == l.getConnections().get(0));
		assertTrue(ccc.get(3).connection == l.getConnections().get(1));

		// Simple MLP
		mlp = NNFactory.mlp(new int[] { 3, 4 }, true);

		ccc = new BreadthFirstOrderStrategy(mlp, mlp.getOutputLayer()).order();
		assertEquals(2, ccc.size(), 0);
		l = mlp.getOutputLayer();
		assertTrue(ccc.get(0).connection == l.getConnections().get(0));
		assertTrue(ccc.get(1).connection == l.getConnections().get(1));

		// CNN
		NeuralNetworkImpl cnn = NNFactory.convNN(new int[][] { { 3, 3, 2 }, { 2, 2, 1, 1, 1, 0, 0 } }, true);

		ccc = new BreadthFirstOrderStrategy(cnn, cnn.getOutputLayer()).order();
		l = cnn.getOutputLayer();
		assertEquals(2, ccc.size(), 0);
		assertTrue(ccc.get(0).connection == l.getConnections().get(0));
		assertTrue(ccc.get(1).connection == l.getConnections().get(1));
	}

	@Test
	@Ignore
	@Deprecated
	public void testDropoutConstruction() {
		NeuralNetworkImpl mlp = CalculationFactory.mlpSigmoid(new int[] { 2, 2, 1 }, false);
		new NNRandomInitializer(new RandomInitializerImpl(-0.5f, 0.5f)).initialize(mlp);

		BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 0f, 0f } }, new float[][] { { 0f } }), null, null, null, 0f, 0f, 0f, 0f, 0.5f, 1, 1, 1);

		LayerCalculatorImpl lc = (LayerCalculatorImpl) mlp.getLayerCalculator();
		Layer hidden = mlp.getLayers().stream().filter(l -> l != mlp.getInputLayer() && l != mlp.getOutputLayer()).findFirst().get();
		assertTrue(OperationsFactory.hasDropout(lc.getConnectionCalculator(hidden)));
		FullyConnected fc = (FullyConnected) mlp.getOutputLayer().getConnections().get(0);
		float w = fc.getWeights().get(0, 0);

		BackPropagationLayerCalculatorImpl bplc = (BackPropagationLayerCalculatorImpl) bpt.getBPLayerCalculator();
		ConnectionCalculatorTensorFunctions tf = (ConnectionCalculatorTensorFunctions) bplc.getConnectionCalculator(hidden);
		assertTrue(tf.getActivationFunctions().stream().filter(f -> OperationsFactory.isMask(f)).findAny().isPresent());

		bpt.train();

		assertTrue(!OperationsFactory.hasDropout(lc.getConnectionCalculator(hidden)));
		assertEquals(w / 2, fc.getWeights().get(0, 0), 0);
	}

	@Test
	@Ignore
	@Deprecated
	public void testReLU() {
		Tensor t = TensorFactory.tensor(2, 2);
		t.set(0.2f, 0, 0);
		t.set(-0.4f, 0, 1);
		t.set(0.6f, 1, 0);
		t.set(-0.3f, 1, 1);

		TensorFunction relu = OperationsFactory.reLUFunction();
		relu.value(t);

		assertEquals(0.2f, t.get(0, 0), 0);
		assertEquals(0f, t.get(0, 1), 0);
		assertEquals(0.6f, t.get(1, 0), 0);
		assertEquals(0f, t.get(1, 1), 0);
	}

	@Ignore
	@Test
	@Deprecated
	public void testSoftMax()
	{
		Tensor m = TensorFactory.tensor(2, 3);
		m.set(1, 0, 0);
		m.set(2, 0, 1);
		m.set(3, 0, 2);
		m.set(4, 1, 0);
		m.set(5, 1, 1);
		m.set(6, 1, 2);

		float sum1 = (float) (Math.exp(m.get(0, 0)) + Math.exp(m.get(0, 1)) + Math.exp(m.get(0, 2)));
		float sum2 = (float) (Math.exp(m.get(1, 0)) + Math.exp(m.get(1, 1)) + Math.exp(m.get(1, 2)));

		TensorFunction sf = OperationsFactory.softmaxFunction();
		sf.value(m);

		assertEquals(Math.exp(1) / sum1, m.get(0, 0), 0.000001);
		assertEquals(Math.exp(2) / sum1, m.get(0, 1), 0.000001);
		assertEquals(Math.exp(3) / sum1, m.get(0, 2), 0.000001);
		assertEquals(Math.exp(4) / sum2, m.get(1, 0), 0.000001);
		assertEquals(Math.exp(5) / sum2, m.get(1, 1), 0.000001);
		assertEquals(Math.exp(6) / sum2, m.get(1, 2), 0.000001);
	}

	@Ignore
	@Test
	@Deprecated
	public void testSoftMax2()
	{
		Tensor m = TensorFactory.tensor(2, 2);
		m.set(6.60573f, 0, 0);
		m.set(-0.56356f, 0, 1);
		m.set(6.60573f, 1, 0);
		m.set(-0.56356f, 1, 1);
		
		float sum = (float) (Math.exp(m.get(0, 0)) + Math.exp(m.get(0, 1)));
		
		TensorFunction sf = OperationsFactory.softmaxFunction();
		sf.value(m);
		
		assertEquals(Math.exp(6.60573f) / sum, m.get(0, 0), 0.000001);
		assertEquals(Math.exp(-0.56356f) / sum, m.get(0, 1), 0.000001);
		assertEquals(Math.exp(6.60573f) / sum, m.get(1, 0), 0.000001);
		assertEquals(Math.exp(-0.56356f) / sum, m.get(1, 1), 0.000001);
	}
	
	@Test
	@Ignore
	@Deprecated
	public void testSoftNegativeLogProbabilty()
	{
		Tensor activation = TensorFactory.tensor(2, 2);
		activation.set(0.999230719826443f, 0, 0);
		activation.set(0.0007692801735570473f, 0, 1);
		activation.set(0.999230719826443f, 1, 0);
		activation.set(0.0007692801735570473f, 1, 1);

		Tensor target = TensorFactory.tensor(2, 2);
		target.set(1, 0, 0);
		target.set(0, 0, 1);
		target.set(1, 1, 0);
		target.set(0, 1, 1);

		LossFunction sf = OperationsFactory.softmaxLoss();
		float error = sf.getLossFunction(activation, target);

		assertEquals(0.0007695762213886436, error / 2, 0.000001);
	}

	@Test
	@Ignore
	@Deprecated
	public void testMSE()
	{
		Tensor activation = TensorFactory.tensor(2, 2);
		Tensor target = TensorFactory.tensor(2, 2);
		Tensor result = TensorFactory.tensor(2, 2);

		activation.set(2f, 0, 0);
		activation.set(4f, 0, 1);
		activation.set(6f, 1, 0);
		activation.set(8f, 1, 1);

		target.set(1f, 0, 0);
		target.set(2f, 0, 1);
		target.set(3f, 1, 0);
		target.set(4f, 1, 1);

		LossFunction lf = OperationsFactory.mse();
		lf.getLossFunctionDerivative(activation, target, result);

		assertEquals(-1, result.get(0, 0), 0);
		assertEquals(-2, result.get(0, 1), 0);
		assertEquals(-3, result.get(1, 0), 0);
		assertEquals(-4, result.get(1, 1), 0);
	}

	@Test
	@Ignore
	@Deprecated
	public void testSoftmaxLoss()
	{
		Tensor activation = TensorFactory.tensor(2, 2);
		Tensor target = TensorFactory.tensor(2, 2);
		Tensor result = TensorFactory.tensor(2, 2);

		activation.set(2f, 0, 0);
		activation.set(4f, 0, 1);
		activation.set(6f, 1, 0);
		activation.set(8f, 1, 1);

		target.set(1f, 0, 0);
		target.set(2f, 0, 1);
		target.set(3f, 1, 0);
		target.set(4f, 1, 1);

		LossFunction lf = OperationsFactory.softmaxLoss();
		lf.getLossFunctionDerivative(activation, target, result);
		
		assertEquals(-1, result.get(0, 0), 0);
		assertEquals(-2, result.get(0, 1), 0);
		assertEquals(-3, result.get(1, 0), 0);
		assertEquals(-4, result.get(1, 1), 0);
	}

	@Ignore
	@Test
	@Deprecated
	public void testNoise()
	{
		Tensor inputOutput = TensorFactory.tensor(20, 20);
		TensorFunction noise = OperationsFactory.noise(inputOutput, 0.5f, -1);
		noise.value(inputOutput);
		boolean hasNoise = false;
		TensorIterator it = inputOutput.iterator();
		while (it.hasNext() && !hasNoise)
		{
			hasNoise = inputOutput.getElements()[it.next()] == -1;
		}

		assertTrue(hasNoise);
	}
}
