package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.junit.Test;

import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiWeightedSumConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLWeightedSumConnectionCalculator;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.util.Environment;

/**
 * Created by chass on 20.11.14.
 */
public class WeightedSumTest extends AbstractTest {


    @Test
    public void testRandomSimpleFF(){
        long seed = 123456789;

        Tensor seqResult = testRandomSimpleFF(Runtime.CPU_SEQ, seed);
        Tensor openclResult = testRandomSimpleFF(Runtime.OPENCL, seed);

        assertTrue(isEqual(seqResult, openclResult));
    }

    private Tensor testRandomSimpleFF(Runtime runtime, long seed){

        configureGlobalRuntimeEnvironment(runtime);

        Random random = new Random(seed);
        Layer inLayer = new Layer();
        Layer outLayer = new Layer();

        Tensor weights = TensorFactory.tensor(2, 3); // 2x3 weights matrix

        FullyConnected fullConnection = new FullyConnected(inLayer, outLayer, TensorFactory.tensor(weights, new int[][]{{0, 0}, {1, 1}}, false));

        Matrix connectionWeightsMatrix = fullConnection.getWeights();
        connectionWeightsMatrix.set(random.nextFloat(), 0, 0);
        connectionWeightsMatrix.set(random.nextFloat(), 0, 1);
        connectionWeightsMatrix.set(random.nextFloat(), 1, 0);
        connectionWeightsMatrix.set(random.nextFloat(), 1, 1);

        List<Connections> connections = new ArrayList<>();
        connections.add(fullConnection);

        NeuralNetworkImpl nn = new NeuralNetworkImpl();
        nn.addConnections(connections.toArray(new Connections[connections.size()]));

        // create value provider which also created tensors for input and output layer for each connection
        // use batchsize 2 in order to run loop 2 times
        ValuesProvider vp = TensorFactory.tensorProvider(nn, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

        // set input values
        Matrix i1 = vp.get(nn.getInputLayer());
        i1.set(random.nextFloat(), 0, 0);
        i1.set(random.nextFloat(), 0, 1);
        i1.set(random.nextFloat(), 1, 0);
        i1.set(random.nextFloat(), 1, 1);

        ConnectionCalculator aws = OperationsFactory.weightedSum();

        if(runtime.equals(Runtime.OPENCL)) {
            assertTrue(aws instanceof OpenCLWeightedSumConnectionCalculator);
        }else{
            assertTrue(aws instanceof AparapiWeightedSumConnectionCalculator);
        }

        aws.calculate(connections, vp, outLayer);

        return vp.get(nn.getOutputLayer());
    }

    @Test
    public void testRandomFFWithBias(){

        long seed = 123456789;

        Tensor seqResult = testRandomFFWithBias(Runtime.CPU_SEQ, seed);
        Tensor openclResult = testRandomFFWithBias(Runtime.OPENCL, seed);

        assertTrue(isEqual(seqResult,openclResult));

    }

    private Tensor testRandomFFWithBias(Runtime runtime, long seed){

        configureGlobalRuntimeEnvironment(runtime);

        Random random = new Random(seed);
        Layer inLayer = new Layer();
        Layer outLayer = new Layer();

        Tensor weights = TensorFactory.tensor(2, 3); // 2x3 weights matrix

        FullyConnected fullConnection = new FullyConnected(inLayer, outLayer, TensorFactory.tensor(weights, new int[][]{{0, 0}, {1, 1}}, false));
        FullyConnected bc = new FullyConnected(new Layer(), outLayer, 1, 2);

        Matrix connectionWeightsMatrix = fullConnection.getWeights();
        connectionWeightsMatrix.set(random.nextFloat(), 0, 0);
        connectionWeightsMatrix.set(random.nextFloat(), 0, 1);
        connectionWeightsMatrix.set(random.nextFloat(), 1, 0);
        connectionWeightsMatrix.set(random.nextFloat(), 1, 1);

        Matrix bcg = bc.getWeights();
        bcg.set(0.1f, 0, 0);
        bcg.set(0.2f, 1, 0);

        List<Connections> connections = new ArrayList<>();
        connections.add(fullConnection);
        connections.add(bc);

        NeuralNetworkImpl nn = new NeuralNetworkImpl();
        nn.addConnections(connections.toArray(new Connections[connections.size()]));

        // create value provider which also created tensors for input and output layer for each connection
        // use batchsize 2 in order to run loop 2 times
        ValuesProvider vp = TensorFactory.tensorProvider(nn, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

        // set input values
        Matrix i1 = vp.get(nn.getInputLayer());
        i1.set(random.nextFloat(), 0, 0);
        i1.set(random.nextFloat(), 0, 1);
        i1.set(random.nextFloat(), 1, 0);
        i1.set(random.nextFloat(), 1, 1);

        Tensor b1 = vp.get(bc.getInputLayer());
        b1.set(1, 0, 0);
        b1.set(1, 1, 0);


        ConnectionCalculator aws = OperationsFactory.weightedSum();

        if(runtime.equals(Runtime.OPENCL)) {
            assertTrue(aws instanceof OpenCLWeightedSumConnectionCalculator);
        }else{
            assertTrue(aws instanceof AparapiWeightedSumConnectionCalculator);
        }

        aws.calculate(connections, vp, outLayer);

        return vp.get(nn.getOutputLayer());
    }


    @Test
    public void testRandomFFCombinedLayers(){
        long seed = 123456789;

        Tensor seqResult = testRandomFFCombinedLayers(Runtime.CPU_SEQ, seed);
        Tensor openclResult = testRandomFFCombinedLayers(Runtime.OPENCL, seed);

        assertTrue(isEqual(seqResult,openclResult));

    }

    private Tensor testRandomFFCombinedLayers(Runtime runtime, long seed){

        configureGlobalRuntimeEnvironment(runtime);

        Random random = new Random(seed);
        Layer inLayer1 = new Layer();
        Layer inLayer2 = new Layer();
        Layer outLayer = new Layer();

        Tensor weights = TensorFactory.tensor(2, 2, 3); // 2x3 weights matrix

        FullyConnected fullConnection1 = new FullyConnected(inLayer1, outLayer, TensorFactory.tensor(weights, new int[][]{{0, 0, 0}, {0, 1, 2}}, true));
        FullyConnected fullConnection2 = new FullyConnected(inLayer2, outLayer, TensorFactory.tensor(weights, new int[][]{{1, 0, 0}, {1, 1, 2}}, true));
        FullyConnected bc = new FullyConnected(new Layer(), outLayer, 1, 2);

        Matrix connectionWeightsMatrix1 = fullConnection1.getWeights();
        connectionWeightsMatrix1.set(random.nextFloat(), 0, 0);
        connectionWeightsMatrix1.set(random.nextFloat(), 0, 1);
        connectionWeightsMatrix1.set(random.nextFloat(), 1, 0);
        connectionWeightsMatrix1.set(random.nextFloat(), 1, 1);

        Matrix connectionWeightsMatrix2 = fullConnection1.getWeights();
        connectionWeightsMatrix2.set(random.nextFloat(), 0, 0);
        connectionWeightsMatrix2.set(random.nextFloat(), 0, 1);
        connectionWeightsMatrix2.set(random.nextFloat(), 1, 0);
        connectionWeightsMatrix2.set(random.nextFloat(), 1, 1);

        Matrix bcg = bc.getWeights();
        bcg.set(0.1f, 0, 0);
        bcg.set(0.2f, 1, 0);

        List<Connections> connections = new ArrayList<>();
        connections.add(fullConnection1);
        connections.add(fullConnection2);
        connections.add(bc);

        NeuralNetworkImpl nn = new NeuralNetworkImpl();
        nn.addConnections(connections.toArray(new Connections[connections.size()]));

        // create value provider which also created tensors for input and output layer for each connection
        // use batchsize 2 in order to run loop 2 times
        ValuesProvider vp = TensorFactory.tensorProvider(nn, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

        // set input values
        Matrix i1 = vp.get(inLayer1);
        i1.set(random.nextFloat(), 0, 0);
        i1.set(random.nextFloat(), 0, 1);
        i1.set(random.nextFloat(), 1, 0);
        i1.set(random.nextFloat(), 1, 1);

        Matrix i2 = vp.get(inLayer2);
        i2.set(random.nextFloat(), 0, 0);
        i2.set(random.nextFloat(), 0, 1);
        i2.set(random.nextFloat(), 1, 0);
        i2.set(random.nextFloat(), 1, 1);

        ConnectionCalculator aws = OperationsFactory.weightedSum();

        if(runtime.equals(Runtime.OPENCL)) {
            assertTrue(aws instanceof OpenCLWeightedSumConnectionCalculator);
        }else{
            assertTrue(aws instanceof AparapiWeightedSumConnectionCalculator);
        }

        aws.calculate(connections, vp, outLayer);

        return vp.get(nn.getOutputLayer());
    }


    @Test
    public void testNominalFF(){
        testWeightedSumFF(Runtime.CPU_SEQ);
        testWeightedSumFF(Runtime.OPENCL);
    }

    private void testWeightedSumFF(Runtime runtime) {

        configureGlobalRuntimeEnvironment(runtime);

        Layer il1 = new Layer();
        Layer ol = new Layer();
        Layer il2 = new Layer();

        // 2 batchsize, 2 output layer size, 3 input layer size (for fullyconnected)
        Tensor weights = TensorFactory.tensor(2, 2, 3);

        FullyConnected c1 = new FullyConnected(il1, ol, TensorFactory.tensor(weights, new int[][]{{0, 0, 0}, {0, 1, 2}}, true));
        FullyConnected c2 = new FullyConnected(il2, ol, TensorFactory.tensor(weights, new int[][]{{1, 0, 0}, {1, 1, 2}}, true));

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

        // value provider creates tensors for the different layer, in this case input and output
        ValuesProvider vp = TensorFactory.tensorProvider(nn, 2, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

        // set inut values
        Matrix i1 = vp.get(nn.getInputLayer());
        i1.set(1, 0, 0);
        i1.set(2, 0, 1);
        i1.set(3, 0, 2);
        i1.set(4, 1, 0);
        i1.set(5, 1, 1);
        i1.set(6, 1, 2);


        ConnectionCalculator aws = OperationsFactory.weightedSum();

        if(runtime.equals(Runtime.OPENCL)){
            assertTrue(aws instanceof OpenCLWeightedSumConnectionCalculator);
        }else{
            assertTrue(aws instanceof AparapiWeightedSumConnectionCalculator);
        }


        // run operaton and input values + weights, result is the output layer
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

        if(runtime.equals(Runtime.OPENCL)){
            assertTrue(aws2 instanceof OpenCLWeightedSumConnectionCalculator);
        }else{
            assertTrue(aws2 instanceof AparapiWeightedSumConnectionCalculator);
        }

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

        if(runtime.equals(Runtime.OPENCL)){
            assertTrue(aws2 instanceof OpenCLWeightedSumConnectionCalculator);
        }else{
            assertTrue(aws2 instanceof AparapiWeightedSumConnectionCalculator);
        }


        aws2.calculate(connections, vp, ol);

        o = vp.get(nn.getOutputLayer());
        assertEquals(28.1, o.get(0, 0), 0.01);
        assertEquals(64.1, o.get(1, 0), 0.01);
        assertEquals(64.2, o.get(0, 1), 0.01);
        assertEquals(154.2, o.get(1, 1), 0.01);
    }

    @Test
    public void testNominalBP(){
        testWeightedSumBP(Runtime.CPU_SEQ);
        testWeightedSumBP(Runtime.OPENCL);
    }

    // test this because weights are accessed in a different order
    private void testWeightedSumBP(Runtime runtime) {

        configureGlobalRuntimeEnvironment(runtime);

        Layer il1 = new Layer();
        Layer ol = new Layer();
        Layer il2 = new Layer();

        Tensor weights = TensorFactory.tensor(2, 3, 2);
        FullyConnected c1 = new FullyConnected(ol, il1, TensorFactory.tensor(weights, new int[][]{{0, 0, 0}, {0, 2, 1}}, true));
        FullyConnected c2 = new FullyConnected(ol, il2, TensorFactory.tensor(weights, new int[][]{{1, 0, 0}, {1, 2, 1}}, true));
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

        if(runtime.equals(Runtime.OPENCL)){
            assertTrue(aws instanceof OpenCLWeightedSumConnectionCalculator);
        }else{
            assertTrue(aws instanceof AparapiWeightedSumConnectionCalculator);
        }

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

        if(runtime.equals(Runtime.OPENCL)){
            assertTrue(aws2 instanceof OpenCLWeightedSumConnectionCalculator);
        }else{
            assertTrue(aws2 instanceof AparapiWeightedSumConnectionCalculator);
        }

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

        if(runtime.equals(Runtime.OPENCL)){
            assertTrue(aws2 instanceof OpenCLWeightedSumConnectionCalculator);
        }else{
            assertTrue(aws2 instanceof AparapiWeightedSumConnectionCalculator);
        }

        aws2.calculate(connections, vp, ol);

        o = vp.get(ol);
        assertEquals(28.1, o.get(0, 0), 0.01);
        assertEquals(64.1, o.get(1, 0), 0.01);
        assertEquals(64.2, o.get(0, 1), 0.01);
        assertEquals(154.2, o.get(1, 1), 0.01);
    }

    @Test
    public void testWeightedSumFFSimple() {
        testWeightedSumFFSimple(Runtime.CPU_SEQ);
        testWeightedSumFFSimple(Runtime.OPENCL);
    }

    private void testWeightedSumFFSimple(Runtime runtime) {

        configureGlobalRuntimeEnvironment(runtime);

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

    @Test
    public void testParallelNetworks() {
        testParallelNetworks(Runtime.CPU_SEQ);
        testParallelNetworks(Runtime.OPENCL);

    }

    private void testParallelNetworks(Runtime runtime)
    {
        configureGlobalRuntimeEnvironment(runtime);

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

}
