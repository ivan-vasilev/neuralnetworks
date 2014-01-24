package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.github.neuralnetworks.architecture.types.DBN;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.StackedAutoencoder;

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
}
