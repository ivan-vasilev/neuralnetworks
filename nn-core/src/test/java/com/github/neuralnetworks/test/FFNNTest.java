package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.BreadthFirstOrderStrategy;
import com.github.neuralnetworks.calculation.LayerOrderStrategy.ConnectionCandidate;
import com.github.neuralnetworks.calculation.TargetLayerOrderStrategy;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSumConnectionCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorFullyConnected;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Util;

/**
 * General feedforward neural networks tests
 */
public class FFNNTest {

    @Test
    public void testWeightedSumFF() {
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.GPU);

	Matrix o = new Matrix(2, 2);

	Layer il1 = new Layer();
	Layer ol = new Layer();
	Layer il2 = new Layer();
	FullyConnected c1 = new FullyConnected(il1, ol, 3, 2);
	FullyConnected c2 = new FullyConnected(il2, ol, 3, 2);
	FullyConnected bc = new FullyConnected(new Layer(), ol, 1, 2);

	Matrix cg = c1.getConnectionGraph();
	cg.set(0, 0, 1);
	cg.set(0, 1, 2);
	cg.set(0, 2, 3);
	cg.set(1, 0, 4);
	cg.set(1, 1, 5);
	cg.set(1, 2, 6);

	cg = c2.getConnectionGraph();
	cg.set(0, 0, 1);
	cg.set(0, 1, 2);
	cg.set(0, 2, 3);
	cg.set(1, 0, 4);
	cg.set(1, 1, 5);
	cg.set(1, 2, 6);

	Matrix i1 = new Matrix(3, 2);
	i1.set(0, 0, 1);
	i1.set(1, 0, 2);
	i1.set(2, 0, 3);
	i1.set(0, 1, 4);
	i1.set(1, 1, 5);
	i1.set(2, 1, 6);

	Matrix i2 = new Matrix(3, 2);
	i2.set(0, 0, 1);
	i2.set(1, 0, 2);
	i2.set(2, 0, 3);
	i2.set(0, 1, 4);
	i2.set(1, 1, 5);
	i2.set(2, 1, 6);

	Matrix bcg = bc.getConnectionGraph();
	bcg.set(0, 0, 0.1f);
	bcg.set(1, 0, 0.2f);

	ConnectionCalculatorFullyConnected aws = new AparapiWeightedSumConnectionCalculator();

	List<Connections> connections = new ArrayList<>();
	connections.add(c1);

	ValuesProvider vp = new ValuesProvider();
	vp.addValues(c1.getInputLayer(), i1);
	vp.addValues(ol, o);

	aws.calculate(connections, vp, ol);

	// most simple case
	assertEquals(14, o.get(0, 0), 0);
	assertEquals(32, o.get(0, 1), 0);
	assertEquals(32, o.get(1, 0), 0);
	assertEquals(77, o.get(1, 1), 0);
	Util.fillArray(o.getElements(), 0);

	// with bias
	connections = new ArrayList<>();
	connections.add(c1);
	connections.add(bc);

	vp = new ValuesProvider();
	vp.addValues(c1.getInputLayer(), i1);
	vp.addValues(ol, o);

	aws = new AparapiWeightedSumConnectionCalculator();
	aws.calculate(connections, vp, ol);

	assertEquals(14.1, o.get(0, 0), 0.01);
	assertEquals(32.1, o.get(0, 1), 0.01);
	assertEquals(32.2, o.get(1, 0), 0.01);
	assertEquals(77.2, o.get(1, 1), 0.01);
	Util.fillArray(o.getElements(), 0);

	// combined layers
	connections = new ArrayList<>();
	connections.add(c1);
	connections.add(c2);
	connections.add(bc);

	vp = new ValuesProvider();
	vp.addValues(c1.getInputLayer(), i1);
	vp.addValues(c2.getInputLayer(), i2);
	vp.addValues(ol, o);

	aws = new AparapiWeightedSumConnectionCalculator();
	aws.calculate(connections, vp, ol);

	assertEquals(28.1, o.get(0, 0), 0.01);
	assertEquals(64.1, o.get(0, 1), 0.01);
	assertEquals(64.2, o.get(1, 0), 0.01);
	assertEquals(154.2, o.get(1, 1), 0.01);
	Util.fillArray(o.getElements(), 0);
    }

    @Test
    public void testWeightedSumBP() {
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.GPU);

	Matrix o = new Matrix(2, 2);

	Layer il1 = new Layer();
	Layer ol = new Layer();
	Layer il2 = new Layer();
	FullyConnected c1 = new FullyConnected(ol, il1, 2, 3);
	FullyConnected c2 = new FullyConnected(ol, il2, 2, 3);
	FullyConnected bc = new FullyConnected(new Layer(), ol, 1, 2);

	Matrix cg = c1.getConnectionGraph();
	cg.set(0, 0, 1);
	cg.set(1, 0, 2);
	cg.set(2, 0, 3);
	cg.set(0, 1, 4);
	cg.set(1, 1, 5);
	cg.set(2, 1, 6);

	cg = c2.getConnectionGraph();
	cg.set(0, 0, 1);
	cg.set(1, 0, 2);
	cg.set(2, 0, 3);
	cg.set(0, 1, 4);
	cg.set(1, 1, 5);
	cg.set(2, 1, 6);

	Matrix i1 = new Matrix(3, 2);
	i1.set(0, 0, 1);
	i1.set(1, 0, 2);
	i1.set(2, 0, 3);
	i1.set(0, 1, 4);
	i1.set(1, 1, 5);
	i1.set(2, 1, 6);

	Matrix i2 = new Matrix(3, 2);
	i2.set(0, 0, 1);
	i2.set(1, 0, 2);
	i2.set(2, 0, 3);
	i2.set(0, 1, 4);
	i2.set(1, 1, 5);
	i2.set(2, 1, 6);

	Matrix bcg = bc.getConnectionGraph();
	bcg.set(0, 0, 0.1f);
	bcg.set(1, 0, 0.2f);

	ConnectionCalculatorFullyConnected aws = new AparapiWeightedSumConnectionCalculator();

	List<Connections> connections = new ArrayList<>();
	connections.add(c1);

	ValuesProvider vp = new ValuesProvider();
	vp.addValues(c1.getOutputLayer(), i1);
	vp.addValues(ol, o);

	aws.calculate(connections, vp, ol);

	// most simple case
	assertEquals(14, o.get(0, 0), 0);
	assertEquals(32, o.get(0, 1), 0);
	assertEquals(32, o.get(1, 0), 0);
	assertEquals(77, o.get(1, 1), 0);
	Util.fillArray(o.getElements(), 0);

	// with bias
	connections = new ArrayList<>();
	connections.add(c1);
	connections.add(bc);

	vp = new ValuesProvider();
	vp.addValues(c1.getOutputLayer(), i1);
	vp.addValues(ol, o);

	aws = new AparapiWeightedSumConnectionCalculator();
	aws.calculate(connections, vp, ol);

	assertEquals(14.1, o.get(0, 0), 0.01);
	assertEquals(32.1, o.get(0, 1), 0.01);
	assertEquals(32.2, o.get(1, 0), 0.01);
	assertEquals(77.2, o.get(1, 1), 0.01);
	Util.fillArray(o.getElements(), 0);

	// combined layers
	connections = new ArrayList<>();
	connections.add(c1);
	connections.add(c2);
	connections.add(bc);

	vp = new ValuesProvider();
	vp.addValues(c1.getOutputLayer(), i1);
	vp.addValues(c2.getOutputLayer(), i2);
	vp.addValues(ol, o);

	aws = new AparapiWeightedSumConnectionCalculator();
	aws.calculate(connections, vp, ol);

	assertEquals(28.1, o.get(0, 0), 0.01);
	assertEquals(64.1, o.get(0, 1), 0.01);
	assertEquals(64.2, o.get(1, 0), 0.01);
	assertEquals(154.2, o.get(1, 1), 0.01);
	Util.fillArray(o.getElements(), 0);
    }

    /**
     * Simple backpropagation test with specific values
     */
    @Test
    public void testSigmoidBP() {
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 2, 2, 1 }, false);

	FullyConnected c1 = (FullyConnected) mlp.getInputLayer().getConnections().iterator().next();
	Matrix cg1 = c1.getConnectionGraph();
	cg1.set(0, 0, 0.1f);
	cg1.set(0, 1, 0.8f);
	cg1.set(1, 0, 0.4f);
	cg1.set(1, 1, 0.6f);

	FullyConnected c2 = (FullyConnected) mlp.getOutputLayer().getConnections().iterator().next();
	Matrix cg2 = c2.getConnectionGraph();
	cg2.set(0, 0, 0.3f);
	cg2.set(0, 1, 0.9f);

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }, 1, 1), new SimpleInputProvider(new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }, 1, 1), null, null, 1f, 0f, 0f, 0f);
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
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 3, 2, 1 }, true);

	List<Connections> c = mlp.getConnections();
	FullyConnected c1 = (FullyConnected) c.get(0);
	Matrix cg1 = c1.getConnectionGraph();
	cg1.set(0, 0, 0.2f);
	cg1.set(0, 1, 0.4f);
	cg1.set(0, 2, -0.5f);
	cg1.set(1, 0, -0.3f);
	cg1.set(1, 1, 0.1f);
	cg1.set(1, 2, 0.2f);

	FullyConnected cb1 = (FullyConnected) c.get(1);
	Matrix cgb1 = cb1.getConnectionGraph();
	cgb1.set(0, 0, -0.4f);
	cgb1.set(0, 1, 0.2f);

	FullyConnected c2 = (FullyConnected) c.get(2);
	Matrix cg2 = c2.getConnectionGraph();
	cg2.set(0, 0, -0.3f);
	cg2.set(0, 1, -0.2f);

	FullyConnected cb2 = (FullyConnected) c.get(3);
	Matrix cgb2 = cb2.getConnectionGraph();
	cgb2.set(0, 0, 0.1f);

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, new SimpleInputProvider(new float[][] { { 1, 0, 1 } }, new float[][] { { 1 } }, 1, 1), new SimpleInputProvider(new float[][] { { 1, 0, 1 } }, new float[][] { { 1 } }, 1, 1), null, null, 0.9f, 0f, 0f, 0f);
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
	assertEquals(0.194, cgb1.get(0, 1), 0.001);

	assertEquals(0.218, cgb2.get(0, 0), 0.001);
    }

    @Test
    public void testParallelNetworks() {
	NeuralNetworkImpl mlp = new NeuralNetworkImpl();
	Layer input = new Layer();
	mlp.addLayer(input);

	Layer leaf1 = new Layer();
	FullyConnected fc1 = new FullyConnected(input, leaf1, 2, 3);
	Util.fillArray(fc1.getConnectionGraph().getElements(), 0.1f);
	mlp.addConnection(fc1);

	Layer leaf2 = new Layer();
	FullyConnected fc2 = new FullyConnected(input, leaf2, 2, 3);
	Util.fillArray(fc2.getConnectionGraph().getElements(), 0.2f);
	mlp.addConnection(fc2);

	Layer output = new Layer();
	FullyConnected fc3 = new FullyConnected(leaf1, output, 3, 1);
	Util.fillArray(fc3.getConnectionGraph().getElements(), 0.3f);
	mlp.addConnection(fc3);
	FullyConnected fc4 = new FullyConnected(leaf2, output, 3, 1);
	Util.fillArray(fc4.getConnectionGraph().getElements(), 0.4f);
	mlp.addConnection(fc4);

	mlp.setLayerCalculator(NNFactory.lcWeightedSum(mlp, null));

	Matrix i = new Matrix(new float [] {2, 2}, 1);
	Set<Layer> calculated = new HashSet<>();
	calculated.add(mlp.getInputLayer());

	ValuesProvider results = new ValuesProvider();
	results.addValues(input, i);

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.SEQ);

	mlp.getLayerCalculator().calculate(mlp, output, calculated, results);

	assertEquals(1.32, results.getValues(output).get(0, 0), 0.000001);
    }

    @Test
    public void testRemoveLayer() {
	NeuralNetworkImpl mlp = NNFactory.mlp(new int[] {3, 4, 5}, true);
	assertEquals(5, mlp.getLayers().size(), 0);
	Layer currentOutput = mlp.getOutputLayer();
	mlp.removeLayer(mlp.getOutputLayer());
	assertEquals(3, mlp.getLayers().size(), 0);
	assertEquals(true, currentOutput != mlp.getOutputLayer());
    }

    @Test
    public void testLayerOrderStrategy() {
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
	NeuralNetworkImpl cnn = NNFactory.convNN(new int[][] { { 3, 3, 2 }, { 2, 2, 1 } }, true);

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
