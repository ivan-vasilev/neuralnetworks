package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import java.util.SortedMap;
import java.util.TreeMap;

import org.junit.Ignore;
import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.BiasLayer;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.MultiLayerPerceptron;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSum;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorFullyConnected;
import com.github.neuralnetworks.test.XorInputProvider.XorOutputError;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.random.AparapiXORShiftInitializer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Util;

/**
 * General feedforward neural networks tests
 */
public class FFNNTest {

    @Test
    public void testWeightedSumFF() {
	Matrix o = new Matrix(2, 2);

	Layer il1 = new Layer(3);
	Layer ol = new Layer(2);
	Layer il2 = new Layer(3);
	FullyConnected c1 = new FullyConnected(il1, ol);
	FullyConnected c2 = new FullyConnected(il2, ol);
	FullyConnected bc = new FullyConnected(new BiasLayer(), ol);

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

	ConnectionCalculatorFullyConnected aws = new ConnectionCalculatorFullyConnected() {

	    private static final long serialVersionUID = 1L;

	    @Override
	    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Matrix> inputConnections, int inputOutputSamples, Layer targetLayer) {
		return new AparapiWeightedSum(inputConnections, inputOutputSamples, targetLayer);
	    }
	};

	TreeMap<Connections, Matrix> map = new TreeMap<Connections, Matrix>();
	map.put(c1, i1);
	aws.calculate(map, o, ol);

	// most simple case
	assertEquals(14, o.get(0, 0), 0);
	assertEquals(32, o.get(0, 1), 0);
	assertEquals(32, o.get(1, 0), 0);
	assertEquals(77, o.get(1, 1), 0);
	Util.fillArray(o.getElements(), 0);

	// with bias
	map.put(bc, new Matrix(2, 2));
	aws = new ConnectionCalculatorFullyConnected() {

	    private static final long serialVersionUID = 1L;

	    @Override
	    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Matrix> inputConnections, int inputOutputSamples, Layer targetLayer) {
		return new AparapiWeightedSum(inputConnections, inputOutputSamples, targetLayer);
	    }
	};
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);
	aws.calculate(map, o, ol);

	assertEquals(14.1, o.get(0, 0), 0.01);
	assertEquals(32.1, o.get(0, 1), 0.01);
	assertEquals(32.2, o.get(1, 0), 0.01);
	assertEquals(77.2, o.get(1, 1), 0.01);
	Util.fillArray(o.getElements(), 0);

	// combined layers
	map.put(c2, i2);
	aws = new ConnectionCalculatorFullyConnected() {

	    private static final long serialVersionUID = 1L;

	    @Override
	    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Matrix> inputConnections, int inputOutputSamples, Layer targetLayer) {
		return new AparapiWeightedSum(inputConnections, inputOutputSamples, targetLayer);
	    }
	};
	aws.calculate(map, o, ol);

	assertEquals(28.1, o.get(0, 0), 0.01);
	assertEquals(64.1, o.get(0, 1), 0.01);
	assertEquals(64.2, o.get(1, 0), 0.01);
	assertEquals(154.2, o.get(1, 1), 0.01);
	Util.fillArray(o.getElements(), 0);
    }

    @Test
    public void testWeightedSumBP() {
	Matrix o = new Matrix(2, 2);

	Layer il1 = new Layer(3);
	Layer ol = new Layer(2);
	Layer il2 = new Layer(3);
	FullyConnected c1 = new FullyConnected(ol, il1);
	FullyConnected c2 = new FullyConnected(ol, il2);
	FullyConnected bc = new FullyConnected(new BiasLayer(), ol);

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

	ConnectionCalculatorFullyConnected aws = new ConnectionCalculatorFullyConnected() {

	    private static final long serialVersionUID = 1L;

	    @Override
	    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Matrix> inputConnections, int inputOutputSamples, Layer targetLayer) {
		return new AparapiWeightedSum(inputConnections, inputOutputSamples, targetLayer);
	    }
	};

	TreeMap<Connections, Matrix> map = new TreeMap<Connections, Matrix>();
	map.put(c1, i1);
	aws.calculate(map, o, ol);

	// most simple case
	assertEquals(14, o.get(0, 0), 0);
	assertEquals(32, o.get(0, 1), 0);
	assertEquals(32, o.get(1, 0), 0);
	assertEquals(77, o.get(1, 1), 0);
	Util.fillArray(o.getElements(), 0);

	// with bias
	map.put(bc, new Matrix(2, 2));
	aws = new ConnectionCalculatorFullyConnected() {

	    private static final long serialVersionUID = 1L;

	    @Override
	    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Matrix> inputConnections, int inputOutputSamples, Layer targetLayer) {
		return new AparapiWeightedSum(inputConnections, inputOutputSamples, targetLayer);
	    }
	};
	aws.calculate(map, o, ol);

	assertEquals(14.1, o.get(0, 0), 0.01);
	assertEquals(32.1, o.get(0, 1), 0.01);
	assertEquals(32.2, o.get(1, 0), 0.01);
	assertEquals(77.2, o.get(1, 1), 0.01);
	Util.fillArray(o.getElements(), 0);

	// combined layers
	map.put(c2, i2);
	aws = new ConnectionCalculatorFullyConnected() {

	    private static final long serialVersionUID = 1L;

	    @Override
	    protected ConnectionCalculator createInputFunction(SortedMap<GraphConnections, Matrix> inputConnections, int inputOutputSamples, Layer targetLayer) {
		return new AparapiWeightedSum(inputConnections, inputOutputSamples, targetLayer);
	    }
	};
	aws.calculate(map, o, ol);

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
	MultiLayerPerceptron mlp = NNFactory.mlpSigmoid(new int[] { 2, 2, 1 }, false);
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

	@SuppressWarnings("unchecked")
	BackPropagationTrainer<MultiLayerPerceptron> bpt = TrainerFactory.backPropagationSigmoid(mlp, new SimpleInputProvider(new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }, 1), new SimpleInputProvider(new float[][] { { 0.35f, 0.9f } }, new float[][] { { 0.5f } }, 1), null, null, 1f,
		0f, 0f);
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);
	bpt.train();

	assertEquals(0.09916, cg1.get(0, 0), 0.01);
	assertEquals(0.7978, cg1.get(0, 1), 0.01);
	assertEquals(0.3972, cg1.get(1, 0), 0.01);
	assertEquals(0.5928, cg1.get(1, 1), 0.01);
	assertEquals(0.272392, cg2.get(0, 0), 0.01);
	assertEquals(0.87305, cg2.get(0, 1), 0.01);
    }

    /**
     * Simple xor backpropagation test
     */
    @Test
    public void testXORSigmoidBP() {
	MultiLayerPerceptron mlp = NNFactory.mlpSigmoid(new int[] { 2, 4, 1 }, true);
	@SuppressWarnings("unchecked")
	BackPropagationTrainer<MultiLayerPerceptron> bpt = TrainerFactory.backPropagationSigmoid(mlp, new XorInputProvider(1000), new XorInputProvider(1), new XorOutputError(), new AparapiXORShiftInitializer(-0.01f, 0.01f), 1f, 0.5f, 0f);
	// Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);
	bpt.train();
	bpt.test();
	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    /**
     * Simple xor backpropagation test
     */
    @Ignore
    @Test
    public void testXORReLUBP() {
	MultiLayerPerceptron mlp = NNFactory.mlpRelu(new int[] { 2, 4, }, true, new AparapiSigmoid());
	mlp.addLayer(new Layer(1), true);
	LayerCalculatorImpl lc = (LayerCalculatorImpl) mlp.getLayerCalculator();
	lc.addConnectionCalculator(mlp.getOutputLayer(), new AparapiSigmoid());

	@SuppressWarnings("unchecked")
	BackPropagationTrainer<MultiLayerPerceptron> bpt = TrainerFactory.backPropagationSigmoid(mlp, new XorInputProvider(1000), new XorInputProvider(1), new XorOutputError(), new AparapiXORShiftInitializer(-0.01f, 0.01f), 1f, 0.5f, 0f);
	// Environment.getInstance().setExecutionMode(EXECUTION_MODE.JTP);
	bpt.train();
	bpt.test();
	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }
}
