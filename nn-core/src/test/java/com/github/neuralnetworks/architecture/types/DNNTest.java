package com.github.neuralnetworks.architecture.types;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.architecture.types.DBN;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.architecture.types.StackedAutoencoder;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.DNNLayerTrainer;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.rbm.CDTrainerBase;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;

@RunWith(Parameterized.class)
public class DNNTest extends AbstractTest
{
	public DNNTest(RuntimeConfiguration conf)
	{
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
	public void testDBNConstruction()
	{
		DBN dbn = NNFactory.dbn(new int[] { 4, 4, 4, 4 }, false);
		assertEquals(4, dbn.getLayers().size(), 0);
		assertEquals(3, dbn.getNeuralNetworks().size(), 0);
		assertEquals(2, dbn.getFirstNeuralNetwork().getLayers().size(), 0);
		assertEquals(2, dbn.getNeuralNetwork(1).getLayers().size(), 0);
		assertEquals(2, dbn.getLastNeuralNetwork().getLayers().size(), 0);

		dbn = NNFactory.dbn(new int[] { 4, 4, 4, 4 }, true);
		assertEquals(7, dbn.getLayers().size(), 0);
		assertEquals(3, dbn.getNeuralNetworks().size(), 0);
		assertEquals(4, dbn.getFirstNeuralNetwork().getLayers().size(), 0);
		assertEquals(4, dbn.getNeuralNetwork(0).getLayers().size(), 0);
		assertEquals(4, dbn.getLastNeuralNetwork().getLayers().size(), 0);

		assertEquals(true, dbn.getFirstNeuralNetwork().getHiddenBiasConnections() != null);
		assertEquals(true, dbn.getFirstNeuralNetwork().getVisibleBiasConnections() != null);
		assertEquals(true, dbn.getNeuralNetwork(1).getHiddenBiasConnections() != null);
		assertEquals(true, dbn.getNeuralNetwork(1).getVisibleBiasConnections() != null);
		assertEquals(true, dbn.getLastNeuralNetwork().getHiddenBiasConnections() != null);
		assertEquals(true, dbn.getLastNeuralNetwork().getVisibleBiasConnections() != null);

		assertEquals(false, dbn.getLayers().contains(dbn.getFirstNeuralNetwork().getVisibleBiasConnections().getInputLayer()));
		assertEquals(false, dbn.getLayers().contains(dbn.getNeuralNetwork(1).getVisibleBiasConnections().getInputLayer()));
		assertEquals(false, dbn.getLayers().contains(dbn.getLastNeuralNetwork().getVisibleBiasConnections().getInputLayer()));

		assertEquals(true, dbn.getFirstNeuralNetwork().getHiddenLayer() == dbn.getNeuralNetwork(1).getVisibleLayer());
		assertEquals(true, dbn.getNeuralNetwork(1).getHiddenLayer() == dbn.getLastNeuralNetwork().getVisibleLayer());

		assertEquals(true, dbn.getOutputLayer().equals(dbn.getLastNeuralNetwork().getHiddenLayer()));
	}

	@Test
	public void testStackedAutoencoderConstruction()
	{
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
	public void testDBNCalculation()
	{
		DBN dbn = NNFactory.dbn(new int[] { 3, 2, 2 }, true);
		dbn.setLayerCalculator(CalculationFactory.lcWeightedSum(dbn, null));

		RBM firstRBM = dbn.getFirstNeuralNetwork();

		Tensor t = firstRBM.getMainConnections().getWeights();
		float[] e1 = t.getElements();
		t.forEach(i -> e1[i] = 0.2f);

		t = firstRBM.getVisibleBiasConnections().getWeights();
		float[] e2 = t.getElements();
		t.forEach(i -> e2[i] = 0.1f);

		t = firstRBM.getHiddenBiasConnections().getWeights();
		float[] e3 = t.getElements();
		t.forEach(i -> e3[i] = 0.3f);

		RBM secondRBM = dbn.getLastNeuralNetwork();

		t = secondRBM.getMainConnections().getWeights();
		float[] e4 = t.getElements();
		t.forEach(i -> e4[i] = 0.4f);

		t = secondRBM.getVisibleBiasConnections().getWeights();
		float[] e5 = t.getElements();
		t.forEach(i -> e5[i] = 0.8f);

		t = secondRBM.getHiddenBiasConnections().getWeights();
		float[] e6 = t.getElements();
		t.forEach(i -> e6[i] = 0.5f);

		Set<Layer> calculatedLayers = new HashSet<>();
		calculatedLayers.add(dbn.getInputLayer());

		ValuesProvider results = TensorFactory.tensorProvider(dbn, 1, true);
		results.get(dbn.getInputLayer()).set(1, 0, 0);
		results.get(dbn.getInputLayer()).set(0, 0, 1);
		results.get(dbn.getInputLayer()).set(1, 0, 2);
		dbn.getLayerCalculator().calculate(dbn, dbn.getOutputLayer(), calculatedLayers, results);

		assertEquals(1.06, results.get(dbn.getOutputLayer()).get(0, 0), 0.00001);
		assertEquals(1.06, results.get(dbn.getOutputLayer()).get(0, 1), 0.00001);
	}

	@Test
	public void testSAECalculation()
	{
		StackedAutoencoder sae = NNFactory.sae(new int[] { 3, 2, 2 }, true);
		sae.setLayerCalculator(CalculationFactory.lcWeightedSum(sae, null));

		Autoencoder firstAE = sae.getFirstNeuralNetwork();
		Tensor t = ((FullyConnected) firstAE.getConnection(firstAE.getInputLayer(), firstAE.getHiddenLayer())).getWeights();
		float[] e1 = t.getElements();
		t.forEach(i -> e1[i] = 0.2f);

		t = ((FullyConnected) firstAE.getConnection(firstAE.getHiddenBiasLayer(), firstAE.getHiddenLayer())).getWeights();
		float[] e2 = t.getElements();
		t.forEach(i -> e2[i] = 0.3f);

		t = ((FullyConnected) firstAE.getConnection(firstAE.getHiddenLayer(), firstAE.getOutputLayer())).getWeights();
		float[] e3 = t.getElements();
		t.forEach(i -> e3[i] = 0.8f);

		t = ((FullyConnected) firstAE.getConnection(firstAE.getOutputBiasLayer(), firstAE.getOutputLayer())).getWeights();
		float[] e4 = t.getElements();
		t.forEach(i -> e4[i] = 0.9f);

		Autoencoder secondAE = sae.getLastNeuralNetwork();

		t = ((FullyConnected) secondAE.getConnection(secondAE.getInputLayer(), secondAE.getHiddenLayer())).getWeights();
		float[] e5 = t.getElements();
		t.forEach(i -> e5[i] = 0.4f);

		t = ((FullyConnected) secondAE.getConnection(secondAE.getHiddenBiasLayer(), secondAE.getHiddenLayer())).getWeights();
		float[] e6 = t.getElements();
		t.forEach(i -> e6[i] = 0.5f);

		t = ((FullyConnected) secondAE.getConnection(secondAE.getHiddenLayer(), secondAE.getOutputLayer())).getWeights();
		float[] e7 = t.getElements();
		t.forEach(i -> e7[i] = 0.7f);

		t = ((FullyConnected) secondAE.getConnection(secondAE.getOutputBiasLayer(), secondAE.getOutputLayer())).getWeights();
		float[] e8 = t.getElements();
		t.forEach(i -> e8[i] = 0.9f);

		Set<Layer> calculatedLayers = new HashSet<>();
		calculatedLayers.add(sae.getInputLayer());

		ValuesProvider results = TensorFactory.tensorProvider(sae, 1, true);
		results.get(sae.getInputLayer()).set(1, 0, 0);
		results.get(sae.getInputLayer()).set(0, 0, 1);
		results.get(sae.getInputLayer()).set(1, 0, 2);

		sae.getLayerCalculator().calculate(sae, sae.getOutputLayer(), calculatedLayers, results);

		assertEquals(1.06, results.get(sae.getOutputLayer()).get(0, 0), 0.00001);
		assertEquals(1.06, results.get(sae.getOutputLayer()).get(0, 1), 0.00001);
	}

	@Test
	public void testDNNLayerTrainer()
	{
		DBN dbn = NNFactory.dbn(new int[] { 3, 2, 2 }, true);
		dbn.setLayerCalculator(CalculationFactory.lcSigmoid(dbn, null));

		RBM firstRBM = dbn.getFirstNeuralNetwork();

		Matrix cg1 = firstRBM.getMainConnections().getWeights();
		cg1.set(0.2f, 0, 0);
		cg1.set(0.4f, 0, 1);
		cg1.set(-0.5f, 0, 2);
		cg1.set(-0.3f, 1, 0);
		cg1.set(0.1f, 1, 1);
		cg1.set(0.2f, 1, 2);

		Matrix cgb1 = firstRBM.getVisibleBiasConnections().getWeights();
		cgb1.set(0f, 0, 0);
		cgb1.set(0f, 1, 0);
		cgb1.set(0f, 2, 0);

		Matrix cgb2 = firstRBM.getHiddenBiasConnections().getWeights();
		cgb2.set(-0.4f, 0, 0);
		cgb2.set(0.2f, 1, 0);

		SimpleInputProvider inputProvider = new SimpleInputProvider(new float[][] { { 1, 0, 1 } });

		CDTrainerBase firstTrainer = TrainerFactory.cdSigmoidTrainer(firstRBM, null, null, null, null, 1f, 0f, 0f, 0f, 1, 1, 1, true);

		RBM secondRBM = dbn.getLastNeuralNetwork();

		CDTrainerBase secondTrainer = TrainerFactory.cdSigmoidTrainer(secondRBM, null, null, null, null, 1f, 0f, 0f, 0f, 1, 1, 1, true);

		Map<NeuralNetwork, OneStepTrainer<?>> layerTrainers = new HashMap<>();
		layerTrainers.put(firstRBM, firstTrainer);
		layerTrainers.put(secondRBM, secondTrainer);

		DNNLayerTrainer trainer = TrainerFactory.dnnLayerTrainer(dbn, layerTrainers, inputProvider, null, null);
		trainer.train();

		assertEquals(0.2 + 0.13203661, cg1.get(0, 0), 0.00001);
		assertEquals(0.4 - 0.22863509, cg1.get(0, 1), 0.00001);
		assertEquals(-0.5 + 0.12887852, cg1.get(0, 2), 0.00001);
		assertEquals(-0.3 + 0.26158813, cg1.get(1, 0), 0.00001);
		assertEquals(0.1 - 0.3014404, cg1.get(1, 1), 0.00001);
		assertEquals(0.2 + 0.25742438, cg1.get(1, 2), 0.00001);

		assertEquals(0.52276707, cgb1.get(0, 0), 0.00001);
		assertEquals(-0.54617375, cgb1.get(1, 0), 0.00001);
		assertEquals(0.51522285, cgb1.get(2, 0), 0.00001);

		assertEquals(-0.4 - 0.08680013, cgb2.get(0, 0), 0.00001);
		assertEquals(0.2 - 0.02693379, cgb2.get(1, 0), 0.00001);
	}

	@Test
	public void testDNNLayerTrainer2()
	{
		DBN dbn = NNFactory.dbn(new int[] { 3, 3, 2 }, true);
		dbn.setLayerCalculator(CalculationFactory.lcSigmoid(dbn, null));

		RBM firstRBM = dbn.getFirstNeuralNetwork();

		LayerCalculatorImpl lc = (LayerCalculatorImpl) dbn.getLayerCalculator();
		lc.addConnectionCalculator(firstRBM.getHiddenLayer(), OperationsFactory.weightedSum());

		Matrix m1 = firstRBM.getMainConnections().getWeights();
		m1.set(1, 0, 0);
		m1.set(0, 0, 1);
		m1.set(0, 0, 2);
		m1.set(0, 1, 0);
		m1.set(1, 1, 1);
		m1.set(0, 1, 2);
		m1.set(0, 2, 0);
		m1.set(0, 2, 1);
		m1.set(1, 2, 2);

		RBM secondRBM = dbn.getLastNeuralNetwork();

		Matrix cg1 = secondRBM.getMainConnections().getWeights();
		cg1.set(0.2f, 0, 0);
		cg1.set(0.4f, 0, 1);
		cg1.set(-0.5f, 0, 2);
		cg1.set(-0.3f, 1, 0);
		cg1.set(0.1f, 1, 1);
		cg1.set(0.2f, 1, 2);

		Matrix cgb1 = secondRBM.getVisibleBiasConnections().getWeights();
		cgb1.set(0f, 0, 0);
		cgb1.set(0f, 1, 0);
		cgb1.set(0f, 2, 0);

		Matrix cgb2 = secondRBM.getHiddenBiasConnections().getWeights();
		cgb2.set(-0.4f, 0, 0);
		cgb2.set(0.2f, 1, 0);

		SimpleInputProvider inputProvider = new SimpleInputProvider(new float[][] { { 1, 0, 1 } });

		CDTrainerBase firstTrainer = TrainerFactory.cdSigmoidTrainer(firstRBM, null, null, null, null, 0f, 0f, 0f, 0f, 0, 1, 1, true);

		CDTrainerBase secondTrainer = TrainerFactory.cdSigmoidTrainer(secondRBM, null, null, null, null, 1f, 0f, 0f, 0f, 1, 1, 1, true);

		Map<NeuralNetwork, OneStepTrainer<?>> layerTrainers = new HashMap<>();
		layerTrainers.put(firstRBM, firstTrainer);
		layerTrainers.put(secondRBM, secondTrainer);

		DNNLayerTrainer trainer = TrainerFactory.dnnLayerTrainer(dbn, layerTrainers, inputProvider, null, null);
		trainer.train();

		assertEquals(0.2 + 0.13203661, cg1.get(0, 0), 0.00001);
		assertEquals(0.4 - 0.22863509, cg1.get(0, 1), 0.00001);
		assertEquals(-0.5 + 0.12887852, cg1.get(0, 2), 0.00001);
		assertEquals(-0.3 + 0.26158813, cg1.get(1, 0), 0.00001);
		assertEquals(0.1 - 0.3014404, cg1.get(1, 1), 0.00001);
		assertEquals(0.2 + 0.25742438, cg1.get(1, 2), 0.00001);

		assertEquals(0.52276707, cgb1.get(0, 0), 0.00001);
		assertEquals(-0.54617375, cgb1.get(1, 0), 0.00001);
		assertEquals(0.51522285, cgb1.get(2, 0), 0.00001);

		assertEquals(-0.4 - 0.08680013, cgb2.get(0, 0), 0.00001);
		assertEquals(0.2 - 0.02693379, cgb2.get(1, 0), 0.00001);
	}
}
