package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.architecture.types.DBN;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.architecture.types.StackedAutoencoder;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSumConnectionCalculator;
import com.github.neuralnetworks.training.DNNLayerTrainer;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.rbm.AparapiCDTrainer;
import com.github.neuralnetworks.util.Util;

public class DNNTest {

    @Test
    public void testDBNConstruction() {
	DBN dbn = NNFactory.dbn(new int[] { 4, 4, 4 }, false);
	assertEquals(3, dbn.getLayers().size(), 0);
	assertEquals(2, dbn.getNeuralNetworks().size(), 0);
	assertEquals(2, dbn.getFirstNeuralNetwork().getLayers().size(), 0);
	assertEquals(2, dbn.getLastNeuralNetwork().getLayers().size(), 0);

	dbn = NNFactory.dbn(new int[] { 4, 4, 4 }, true);
	assertEquals(5, dbn.getLayers().size(), 0);
	assertEquals(2, dbn.getNeuralNetworks().size(), 0);
	assertEquals(4, dbn.getFirstNeuralNetwork().getLayers().size(), 0);
	assertEquals(4, dbn.getLastNeuralNetwork().getLayers().size(), 0);

	assertEquals(true, dbn.getFirstNeuralNetwork().getHiddenBiasConnections() != null);
	assertEquals(true, dbn.getFirstNeuralNetwork().getVisibleBiasConnections() != null);
	assertEquals(true, dbn.getLastNeuralNetwork().getHiddenBiasConnections() != null);
	assertEquals(true, dbn.getLastNeuralNetwork().getVisibleBiasConnections() != null);

	assertEquals(false, dbn.getLayers().contains(dbn.getFirstNeuralNetwork().getVisibleBiasConnections().getInputLayer()));
	assertEquals(false, dbn.getLayers().contains(dbn.getLastNeuralNetwork().getVisibleBiasConnections().getInputLayer()));

	assertEquals(true, dbn.getFirstNeuralNetwork().getHiddenLayer() == dbn.getLastNeuralNetwork().getVisibleLayer());

	assertEquals(true, dbn.getOutputLayer().equals(dbn.getLastNeuralNetwork().getHiddenLayer()));
    }

    @Test
    public void testStackedAutoencoderConstruction() {
	StackedAutoencoder sae = NNFactory.sae(new int[] { 5, 4, 3 }, false);
	assertEquals(3, sae.getLayers().size(), 0);
	assertEquals(2, sae.getNeuralNetworks().size(), 0);
	assertEquals(3, sae.getFirstNeuralNetwork().getLayers().size(), 0);
	assertEquals(3, sae.getLastNeuralNetwork().getLayers().size(), 0);

	sae = NNFactory.sae(new int[] { 5, 4, 3 }, true);
	assertEquals(5, sae.getLayers().size(), 0);
	assertEquals(2, sae.getNeuralNetworks().size(), 0);
	assertEquals(5, sae.getFirstNeuralNetwork().getLayers().size(), 0);
	assertEquals(5, sae.getLastNeuralNetwork().getLayers().size(), 0);

	assertEquals(false, sae.getLayers().contains(sae.getFirstNeuralNetwork().getOutputLayer()));
	assertEquals(false, sae.getLayers().contains(sae.getLastNeuralNetwork().getOutputLayer()));
	assertEquals(true, sae.getOutputLayer().equals(sae.getLastNeuralNetwork().getHiddenLayer()));

	assertEquals(true, sae.getFirstNeuralNetwork().getHiddenLayer() == sae.getLastNeuralNetwork().getInputLayer());
    }

    @Test
    public void testDBNCalculation() {
	DBN dbn = NNFactory.dbn(new int [] {3, 2, 2}, true);
	NNFactory.nnWeightedSum(dbn, null);

	RBM firstRBM = dbn.getFirstNeuralNetwork();
	Util.fillArray(firstRBM.getMainConnections().getConnectionGraph().getElements(), 0.2f);
	Util.fillArray(firstRBM.getVisibleBiasConnections().getConnectionGraph().getElements(), 0.1f);
	Util.fillArray(firstRBM.getHiddenBiasConnections().getConnectionGraph().getElements(), 0.3f);

	RBM secondRBM = dbn.getLastNeuralNetwork();
	Util.fillArray(secondRBM.getMainConnections().getConnectionGraph().getElements(), 0.4f);
	Util.fillArray(secondRBM.getVisibleBiasConnections().getConnectionGraph().getElements(), 0.8f);
	Util.fillArray(secondRBM.getHiddenBiasConnections().getConnectionGraph().getElements(), 0.5f);

	Set<Layer> calculatedLayers = new HashSet<>();
	calculatedLayers.add(dbn.getInputLayer());

	Map<Layer, Matrix> results = new HashMap<>();
	results.put(dbn.getInputLayer(), new Matrix(new float[] {1, 0, 1}, 1));
	dbn.getLayerCalculator().calculate(dbn, dbn.getOutputLayer(), calculatedLayers, results);

	assertEquals(1.06, results.get(dbn.getOutputLayer()).get(0, 0), 0.00001);
	assertEquals(1.06, results.get(dbn.getOutputLayer()).get(1, 0), 0.00001);
    }

    @Test
    public void testSAECalculation() {
	StackedAutoencoder sae = NNFactory.sae(new int [] {3, 2, 2}, true);
	NNFactory.nnWeightedSum(sae, null);

	Autoencoder firstAE = sae.getFirstNeuralNetwork();
	Util.fillArray(((GraphConnections) firstAE.getConnection(firstAE.getInputLayer(), firstAE.getHiddenLayer())).getConnectionGraph().getElements(), 0.2f);
	Util.fillArray(((GraphConnections) firstAE.getConnection(firstAE.getHiddenBiasLayer(), firstAE.getHiddenLayer())).getConnectionGraph().getElements(), 0.3f);
	Util.fillArray(((GraphConnections) firstAE.getConnection(firstAE.getHiddenLayer(), firstAE.getOutputLayer())).getConnectionGraph().getElements(), 0.8f);
	Util.fillArray(((GraphConnections) firstAE.getConnection(firstAE.getOutputBiasLayer(), firstAE.getOutputLayer())).getConnectionGraph().getElements(), 0.9f);

	Autoencoder secondAE = sae.getLastNeuralNetwork();
	Util.fillArray(((GraphConnections) secondAE.getConnection(secondAE.getInputLayer(), secondAE.getHiddenLayer())).getConnectionGraph().getElements(), 0.4f);
	Util.fillArray(((GraphConnections) secondAE.getConnection(secondAE.getHiddenBiasLayer(), secondAE.getHiddenLayer())).getConnectionGraph().getElements(), 0.5f);
	Util.fillArray(((GraphConnections) secondAE.getConnection(secondAE.getHiddenLayer(), secondAE.getOutputLayer())).getConnectionGraph().getElements(), 0.7f);
	Util.fillArray(((GraphConnections) secondAE.getConnection(secondAE.getOutputBiasLayer(), secondAE.getOutputLayer())).getConnectionGraph().getElements(), 0.9f);

	Set<Layer> calculatedLayers = new HashSet<>();
	calculatedLayers.add(sae.getInputLayer());

	Map<Layer, Matrix> results = new HashMap<>();
	results.put(sae.getInputLayer(), new Matrix(new float[] {1, 0, 1}, 1));
	sae.getLayerCalculator().calculate(sae, sae.getOutputLayer(), calculatedLayers, results);

	assertEquals(1.06, results.get(sae.getOutputLayer()).get(0, 0), 0.00001);
	assertEquals(1.06, results.get(sae.getOutputLayer()).get(1, 0), 0.00001);
    }

    @Test
    public void testDNNLayerTrainer() {
	DBN dbn = NNFactory.dbn(new int [] {3, 2, 2}, true);
	NNFactory.nnSigmoid(dbn, null);

	RBM firstRBM = dbn.getFirstNeuralNetwork();

	Matrix cg1 = firstRBM.getMainConnections().getConnectionGraph();
	cg1.set(0, 0, 0.2f);
	cg1.set(0, 1, 0.4f);
	cg1.set(0, 2, -0.5f);
	cg1.set(1, 0, -0.3f);
	cg1.set(1, 1, 0.1f);
	cg1.set(1, 2, 0.2f);

	Matrix cgb1 = firstRBM.getVisibleBiasConnections().getConnectionGraph();
	cgb1.set(0, 0, 0f);
	cgb1.set(1, 0, 0f);
	cgb1.set(2, 0, 0f);

	Matrix cgb2 = firstRBM.getHiddenBiasConnections().getConnectionGraph();
	cgb2.set(0, 0, -0.4f);
	cgb2.set(1, 0, 0.2f);

	SimpleInputProvider inputProvider = new SimpleInputProvider(new float[][] { { 1, 0, 1 } }, null, 1, 1);

	AparapiCDTrainer firstTrainer = TrainerFactory.pcdTrainer(firstRBM, NNFactory.rbmSigmoidSigmoid(firstRBM), null, null, null, null, 1f, 0f, 0f, 1);

	RBM secondRBM = dbn.getLastNeuralNetwork();

	AparapiCDTrainer secondTrainer = TrainerFactory.pcdTrainer(secondRBM, NNFactory.rbmSigmoidSigmoid(secondRBM), null, null, null, null, 1f, 0f, 0f, 1);

	Map<NeuralNetwork, OneStepTrainer<?>> layerTrainers = new HashMap<>();
	layerTrainers.put(firstRBM, firstTrainer);
	layerTrainers.put(secondRBM, secondTrainer);

	DNNLayerTrainer trainer = TrainerFactory.dnnLayerTrainer(dbn, layerTrainers, inputProvider, null, null);
	trainer.train();
	
	assertEquals(0.2 + 0.13203661, cg1.get(0, 0), 0.00001);
	assertEquals(0.4 - 0.22863509,  cg1.get(0, 1), 0.00001);
	assertEquals(-0.5 + 0.12887852, cg1.get(0, 2), 0.00001);
	assertEquals(-0.3 + 0.26158813, cg1.get(1, 0), 0.00001);
	assertEquals(0.1 - 0.3014404,  cg1.get(1, 1), 0.00001);
	assertEquals(0.2 + 0.25742438, cg1.get(1, 2), 0.00001);

	assertEquals(0.52276707, cgb1.get(0, 0), 0.00001);
	assertEquals(- 0.54617375, cgb1.get(1, 0), 0.00001);
	assertEquals(0.51522285, cgb1.get(2, 0), 0.00001);

	assertEquals(-0.4 - 0.08680013, cgb2.get(0, 0), 0.00001);
	assertEquals(0.2 - 0.02693379, cgb2.get(1, 0), 0.00001);
    }

    @Test
    public void testDNNLayerTrainer2() {
	DBN dbn = NNFactory.dbn(new int [] {3, 3, 2}, true);
	NNFactory.nnSigmoid(dbn, null);

	LayerCalculatorImpl lc = (LayerCalculatorImpl) dbn.getLayerCalculator();
	RBM firstRBM = dbn.getFirstNeuralNetwork();
	lc.addConnectionCalculator(firstRBM.getHiddenLayer(), new AparapiWeightedSumConnectionCalculator());

	Matrix m1 = firstRBM.getMainConnections().getConnectionGraph();
	m1.set(0, 0, 1);
	m1.set(0, 1, 0);
	m1.set(0, 2, 0);
	m1.set(1, 0, 0);
	m1.set(1, 1, 1);
	m1.set(1, 2, 0);
	m1.set(2, 0, 0);
	m1.set(2, 1, 0);
	m1.set(2, 2, 1);

	RBM secondRBM = dbn.getLastNeuralNetwork();
	
	Matrix cg1 = secondRBM.getMainConnections().getConnectionGraph();
	cg1.set(0, 0, 0.2f);
	cg1.set(0, 1, 0.4f);
	cg1.set(0, 2, -0.5f);
	cg1.set(1, 0, -0.3f);
	cg1.set(1, 1, 0.1f);
	cg1.set(1, 2, 0.2f);
	
	Matrix cgb1 = secondRBM.getVisibleBiasConnections().getConnectionGraph();
	cgb1.set(0, 0, 0f);
	cgb1.set(1, 0, 0f);
	cgb1.set(2, 0, 0f);
	
	Matrix cgb2 = secondRBM.getHiddenBiasConnections().getConnectionGraph();
	cgb2.set(0, 0, -0.4f);
	cgb2.set(1, 0, 0.2f);
	
	SimpleInputProvider inputProvider = new SimpleInputProvider(new float[][] { { 1, 0, 1 } }, null, 1, 1);

	AparapiCDTrainer firstTrainer = TrainerFactory.pcdTrainer(firstRBM, NNFactory.rbmSigmoidSigmoid(firstRBM), null, null, null, null, 0f, 0f, 0f, 0);

	AparapiCDTrainer secondTrainer = TrainerFactory.pcdTrainer(secondRBM, NNFactory.rbmSigmoidSigmoid(secondRBM), null, null, null, null, 1f, 0f, 0f, 1);

	Map<NeuralNetwork, OneStepTrainer<?>> layerTrainers = new HashMap<>();
	layerTrainers.put(firstRBM, firstTrainer);
	layerTrainers.put(secondRBM, secondTrainer);
	
	DNNLayerTrainer trainer = TrainerFactory.dnnLayerTrainer(dbn, layerTrainers, inputProvider, null, null);
	trainer.train();
	
	assertEquals(0.2 + 0.13203661, cg1.get(0, 0), 0.00001);
	assertEquals(0.4 - 0.22863509,  cg1.get(0, 1), 0.00001);
	assertEquals(-0.5 + 0.12887852, cg1.get(0, 2), 0.00001);
	assertEquals(-0.3 + 0.26158813, cg1.get(1, 0), 0.00001);
	assertEquals(0.1 - 0.3014404,  cg1.get(1, 1), 0.00001);
	assertEquals(0.2 + 0.25742438, cg1.get(1, 2), 0.00001);
	
	assertEquals(0.52276707, cgb1.get(0, 0), 0.00001);
	assertEquals(- 0.54617375, cgb1.get(1, 0), 0.00001);
	assertEquals(0.51522285, cgb1.get(2, 0), 0.00001);
	
	assertEquals(-0.4 - 0.08680013, cgb2.get(0, 0), 0.00001);
	assertEquals(0.2 - 0.02693379, cgb2.get(1, 0), 0.00001);
    }
}
