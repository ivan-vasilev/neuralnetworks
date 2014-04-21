package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.BreadthFirstOrderStrategy;
import com.github.neuralnetworks.calculation.LayerOrderStrategy.ConnectionCandidate;
import com.github.neuralnetworks.calculation.TargetLayerOrderStrategy;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSumConnectionCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorFullyConnected;
import com.github.neuralnetworks.calculation.neuronfunctions.MaxoutWinners;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Matrix;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.TensorFactory;

/**
 * General feedforward neural networks tests
 */
public class FFNNTest {

    @Test
    public void testWeightedSumFF() {
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	Layer il1 = new Layer();
	Layer ol = new Layer();
	Layer il2 = new Layer();

	Tensor weights = TensorFactory.tensor(2, 2, 3);

	FullyConnected c1 = new FullyConnected(il1, ol, TensorFactory.tensor(weights, new int[][]{{0, 0, 0}, {0, 1, 2}}));
	FullyConnected c2 = new FullyConnected(il2, ol, TensorFactory.tensor(weights, new int[][]{{1, 0, 0}, {1, 1, 2}}));
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

	ValuesProvider vp = TensorFactory.tensorProvider(nn, 2, true);

	Matrix i1 = vp.get(nn.getInputLayer());
	i1.set(1, 0, 0);
	i1.set(2, 1, 0);
	i1.set(3, 2, 0);
	i1.set(4, 0, 1);
	i1.set(5, 1, 1);
	i1.set(6, 2, 1);

	ConnectionCalculatorFullyConnected aws = new AparapiWeightedSumConnectionCalculator();
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
	vp = TensorFactory.tensorProvider(nn, 2, true);

	i1 = vp.get(nn.getInputLayer());
	i1.set(1, 0, 0);
	i1.set(2, 1, 0);
	i1.set(3, 2, 0);
	i1.set(4, 0, 1);
	i1.set(5, 1, 1);
	i1.set(6, 2, 1);

	aws = new AparapiWeightedSumConnectionCalculator();
	aws.calculate(connections, vp, ol);

	o = vp.get(nn.getOutputLayer());
	assertEquals(14.1, o.get(0, 0), 0.01);
	assertEquals(32.1, o.get(0, 1), 0.01);
	assertEquals(32.2, o.get(1, 0), 0.01);
	assertEquals(77.2, o.get(1, 1), 0.01);

	// combined layers
	connections = new ArrayList<>();
	connections.add(c1);
	connections.add(c2);
	connections.add(bc);
	nn = new NeuralNetworkImpl();
	nn.addConnections(connections.toArray(new Connections[connections.size()]));
	vp = TensorFactory.tensorProvider(nn, 2, true);

	i1 = vp.get(il1);
	i1.set(1, 0, 0);
	i1.set(2, 1, 0);
	i1.set(3, 2, 0);
	i1.set(4, 0, 1);
	i1.set(5, 1, 1);
	i1.set(6, 2, 1);

	Matrix i2 = vp.get(il2);
	i2.set(1, 0, 0);
	i2.set(2, 1, 0);
	i2.set(3, 2, 0);
	i2.set(4, 0, 1);
	i2.set(5, 1, 1);
	i2.set(6, 2, 1);

	aws = new AparapiWeightedSumConnectionCalculator();
	aws.calculate(connections, vp, ol);

	o = vp.get(nn.getOutputLayer());
	assertEquals(28.1, o.get(0, 0), 0.01);
	assertEquals(64.1, o.get(0, 1), 0.01);
	assertEquals(64.2, o.get(1, 0), 0.01);
	assertEquals(154.2, o.get(1, 1), 0.01);
    }

    @Test
    public void testWeightedSumBP() {
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.GPU);

	Layer il1 = new Layer();
	Layer ol = new Layer();
	Layer il2 = new Layer();

	Tensor weights = TensorFactory.tensor(2, 3, 2);
	FullyConnected c1 = new FullyConnected(ol, il1, TensorFactory.tensor(weights, new int[][]{{0, 0, 0}, {0, 2, 1}}));
	FullyConnected c2 = new FullyConnected(ol, il2, TensorFactory.tensor(weights, new int[][]{{1, 0, 0}, {1, 2, 1}}));
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

	ConnectionCalculatorFullyConnected aws = new AparapiWeightedSumConnectionCalculator();

	List<Connections> connections = new ArrayList<>();
	connections.add(c1);
	NeuralNetworkImpl nn = new NeuralNetworkImpl();
	nn.addConnections(connections.toArray(new Connections[connections.size()]));
	ValuesProvider vp = TensorFactory.tensorProvider(nn, 2, true);

	Matrix i1 = vp.get(il1);
	i1.set(1, 0, 0);
	i1.set(2, 1, 0);
	i1.set(3, 2, 0);
	i1.set(4, 0, 1);
	i1.set(5, 1, 1);
	i1.set(6, 2, 1);

	aws.calculate(connections, vp, ol);

	// most simple case
	Matrix o = vp.get(ol);
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
	vp = TensorFactory.tensorProvider(nn, 2, true);
	i1 = vp.get(il1);
	i1.set(1, 0, 0);
	i1.set(2, 1, 0);
	i1.set(3, 2, 0);
	i1.set(4, 0, 1);
	i1.set(5, 1, 1);
	i1.set(6, 2, 1);

	aws = new AparapiWeightedSumConnectionCalculator();
	aws.calculate(connections, vp, ol);

	o = vp.get(ol);
	assertEquals(14.1, o.get(0, 0), 0.01);
	assertEquals(32.1, o.get(0, 1), 0.01);
	assertEquals(32.2, o.get(1, 0), 0.01);
	assertEquals(77.2, o.get(1, 1), 0.01);

	// combined layers
	connections = new ArrayList<>();
	connections.add(c1);
	connections.add(c2);
	connections.add(bc);
	nn = new NeuralNetworkImpl();
	nn.addConnections(connections.toArray(new Connections[connections.size()]));
	vp = TensorFactory.tensorProvider(nn, 2, true);

	i1 = vp.get(il1);
	i1.set(1, 0, 0);
	i1.set(2, 1, 0);
	i1.set(3, 2, 0);
	i1.set(4, 0, 1);
	i1.set(5, 1, 1);
	i1.set(6, 2, 1);

	Matrix i2 = vp.get(il2);
	i2.set(1, 0, 0);
	i2.set(2, 1, 0);
	i2.set(3, 2, 0);
	i2.set(4, 0, 1);
	i2.set(5, 1, 1);
	i2.set(6, 2, 1);

	aws = new AparapiWeightedSumConnectionCalculator();
	aws.calculate(connections, vp, ol);

	o = vp.get(ol);
	assertEquals(28.1, o.get(0, 0), 0.01);
	assertEquals(64.1, o.get(0, 1), 0.01);
	assertEquals(64.2, o.get(1, 0), 0.01);
	assertEquals(154.2, o.get(1, 1), 0.01);
    }

    /**
     * Simple backpropagation test with specific values
     */
    @Test
    public void testSigmoidBP() {
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);
	Environment.getInstance().setUseWeightsSharedMemory(true);
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 2, 2, 1 }, false);

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

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }), new SimpleInputProvider(new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }), null, null, 1f, 0f, 0f, 0f, 0f, 1, 1, 1);
	bpt.train();

	assertEquals(0.09916, cg1.get(0, 0), 0.01);
	assertEquals(0.7978, cg1.get(0, 1), 0.01);
	assertEquals(0.3972, cg1.get(1, 0), 0.01);
	assertEquals(0.5928, cg1.get(1, 1), 0.01);
	assertEquals(0.272392, cg2.get(0, 0), 0.01);
	assertEquals(0.87305, cg2.get(0, 1), 0.01);
    }

    /**
     * Simple backpropagation test with specific values
     */
    @Test
    public void testSigmoidBP2() {
	//Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);
	Environment.getInstance().setUseWeightsSharedMemory(true);
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 3, 2, 1 }, true);

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

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 1, 0, 1 } }, new float[][] { { 1 } }), new SimpleInputProvider(new float[][] { { 1, 0, 1 } }, new float[][] { { 1 } }), null, null, 0.9f, 0f, 0f, 0f, 0f, 1, 1, 1);
	bpt.train();

	assertEquals(0.192, cg1.get(0, 0), 0.001);
	assertEquals(0.4, cg1.get(0, 1), 0.001);
	assertEquals(-0.508, cg1.get(0, 2), 0.001);
	assertEquals(-0.306, cg1.get(1, 0), 0.001);
	assertEquals(0.1, cg1.get(1, 1), 0.001);
	assertEquals(0.194, cg1.get(1, 2), 0.001);

	assertEquals(-0.261, cg2.get(0, 0), 0.001);
	assertEquals(-0.138, cg2.get(0, 1), 0.001);

	assertEquals(-0.408, cgb1.get(0, 0), 0.001);
	assertEquals(0.194, cgb1.get(1, 0), 0.001);

	assertEquals(0.218, cgb2.get(0, 0), 0.001);
    }

    /**
     * BP with dropout
     */
    @Test
    public void testSigmoidBPDropout() {
	//Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);
	Environment.getInstance().setUseWeightsSharedMemory(true);
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 3, 2, 1 }, true);

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

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 1, 0, 1 } }, new float[][] { { 1 } }), new SimpleInputProvider(new float[][] { { 1, 0, 1 } }, new float[][] { { 1 } }), null, null, 0.9f, 0f, 0f, 0f, 0.01f, 1, 1, 1);
	bpt.train();

	assertEquals(0.192, cg1.get(0, 0), 0.001);
	assertEquals(0.4, cg1.get(0, 1), 0.001);
	assertEquals(-0.508, cg1.get(0, 2), 0.001);
	assertEquals(-0.306, cg1.get(1, 0), 0.001);
	assertEquals(0.1, cg1.get(1, 1), 0.001);
	assertEquals(0.194, cg1.get(1, 2), 0.001);

	assertEquals(-0.261 * 0.99, cg2.get(0, 0), 0.001);
	assertEquals(-0.138 * 0.99, cg2.get(0, 1), 0.001);

	assertEquals(-0.408, cgb1.get(0, 0), 0.001);
	assertEquals(0.194, cgb1.get(1, 0), 0.001);

	assertEquals(0.218, cgb2.get(0, 0), 0.001);
    }

    /**
     * maxout ff
     */
    @Test
    public void testMaxoutFF() {
	//Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);
	Environment.getInstance().setUseWeightsSharedMemory(true);
	NeuralNetworkImpl nn = NNFactory.maxout(new int[] { 2, 2 }, true, null);

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
	int startIndex = MaxoutWinners.getInstance().getStartPositions(Arrays.asList(new Connections[] {c.get(0)}))[0];
	assertEquals(1, winners[startIndex], 0);
	assertEquals(1, winners[startIndex + 1], 0);
    }

    @Test
    public void testSigmoidBP3() {
	Environment.getInstance().setUseDataSharedMemory(true);
	Environment.getInstance().setUseWeightsSharedMemory(true);

	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 3, 2, 1 }, true);

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

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 1, 0, 1 }, { 1, 1, 0 } }, new float[][] { { 1 }, { 1 } }), null, null, null, 0.9f, 0f, 0f, 0f, 0f, 1, 1, 1);
	bpt.train();

	assertEquals(0.1849, cg1.get(0, 0), 0.0001);
	assertEquals(0.3927, cg1.get(0, 1), 0.0001);
	assertEquals(-0.508, cg1.get(0, 2), 0.001);
	assertEquals(-0.3098, cg1.get(1, 0), 0.0001);
	assertEquals(0.0961, cg1.get(1, 1), 0.0001);
	assertEquals(0.194, cg1.get(1, 2), 0.001);

	assertEquals(-0.1996, cg2.get(0, 0), 0.0001);
	assertEquals(-0.0823, cg2.get(0, 1), 0.0001);

	assertEquals(-0.4151, cgb1.get(0, 0), 0.0001);
	assertEquals(0.1902, cgb1.get(1, 0), 0.0001);

	assertEquals(0.3302, cgb2.get(0, 0), 0.0001);
    }

    @Test
    public void testParallelNetworks() {
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	Environment.getInstance().setUseWeightsSharedMemory(true);
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

	mlp.setLayerCalculator(NNFactory.lcWeightedSum(mlp, null));

	Set<Layer> calculated = new HashSet<>();
	calculated.add(mlp.getInputLayer());

	ValuesProvider results = TensorFactory.tensorProvider(mlp, 1, true);
	results.get(mlp.getInputLayer()).set(2, 0, 0);
	results.get(mlp.getInputLayer()).set(2, 1, 0);

	mlp.getLayerCalculator().calculate(mlp, output, calculated, results);

	assertEquals(1.32, results.get(output).get(0, 0), 0.000001);
    }

    @Test
    public void testRemoveLayer() {
	Environment.getInstance().setUseWeightsSharedMemory(true);
	NeuralNetworkImpl mlp = NNFactory.mlp(new int[] {3, 4, 5}, true);
	assertEquals(5, mlp.getLayers().size(), 0);
	Layer currentOutput = mlp.getOutputLayer();
	mlp.removeLayer(mlp.getOutputLayer());
	assertEquals(3, mlp.getLayers().size(), 0);
	assertEquals(true, currentOutput != mlp.getOutputLayer());
    }

    @Test
    public void testLayerOrderStrategy() {
	Environment.getInstance().setUseWeightsSharedMemory(true);

	// MLP
	NeuralNetworkImpl mlp = NNFactory.mlp(new int[] {3, 4, 5}, true);
	
	Set<Layer> calculated = new HashSet<Layer>();
	calculated.add(mlp.getInputLayer());
	List<ConnectionCandidate> ccc = new TargetLayerOrderStrategy(mlp, mlp.getOutputLayer(), calculated).order();
	assertEquals(4, ccc.size(), 0);
	Layer l = mlp.getInputLayer();
	assertTrue(ccc.get(0).connection == l.getConnections().get(0));
	l = l.getConnections().get(0).getOutputLayer();
	assertTrue(ccc.get(1).connection == l.getConnections().get(1));
	assertTrue(ccc.get(2).connection == l.getConnections().get(2));
	l = l.getConnections().get(2).getOutputLayer();
	assertTrue(ccc.get(3).connection == l.getConnections().get(1));

	ccc = new BreadthFirstOrderStrategy(mlp, mlp.getOutputLayer()).order();
	assertEquals(4, ccc.size(), 0);
	l = mlp.getOutputLayer();
	assertTrue(ccc.get(0).connection == l.getConnections().get(0));
	assertTrue(ccc.get(1).connection == l.getConnections().get(1));

	l = l.getConnections().get(0).getInputLayer();
	assertTrue(ccc.get(2).connection == l.getConnections().get(0));
	assertTrue(ccc.get(3).connection == l.getConnections().get(1));

	// Simple MLP
	mlp = NNFactory.mlp(new int[] {3, 4}, true);

	calculated = new HashSet<Layer>();
	calculated.add(mlp.getInputLayer());
	ccc = new TargetLayerOrderStrategy(mlp, mlp.getOutputLayer(), calculated).order();
	assertEquals(2, ccc.size(), 0);
	l = mlp.getOutputLayer();
	assertTrue(ccc.get(0).connection == l.getConnections().get(0));
	assertTrue(ccc.get(1).connection == l.getConnections().get(1));

	ccc = new BreadthFirstOrderStrategy(mlp, mlp.getOutputLayer()).order();
	assertEquals(2, ccc.size(), 0);
	l = mlp.getOutputLayer();
	assertTrue(ccc.get(0).connection == l.getConnections().get(0));
	assertTrue(ccc.get(1).connection == l.getConnections().get(1));

	// CNN
	NeuralNetworkImpl cnn = NNFactory.convNN(new int[][] { { 3, 3, 2 }, { 2, 2, 1, 1 } }, true);

	calculated = new HashSet<Layer>();
	calculated.add(cnn.getInputLayer());
	ccc = new TargetLayerOrderStrategy(cnn, cnn.getOutputLayer(), calculated).order();
	l = cnn.getOutputLayer();
	assertEquals(2, ccc.size(), 0);
	assertTrue(ccc.get(0).connection == l.getConnections().get(0));
	assertTrue(ccc.get(1).connection == l.getConnections().get(1));

	ccc = new BreadthFirstOrderStrategy(cnn, cnn.getOutputLayer()).order();
	l = cnn.getOutputLayer();
	assertEquals(2, ccc.size(), 0);
	assertTrue(ccc.get(0).connection == l.getConnections().get(0));
	assertTrue(ccc.get(1).connection == l.getConnections().get(1));
    }
}
