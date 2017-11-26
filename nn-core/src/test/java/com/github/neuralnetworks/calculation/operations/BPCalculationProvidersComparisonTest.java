package com.github.neuralnetworks.calculation.operations;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.input.RandomInputProvider;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.tensor.Tensor.TensorIterator;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationLayerCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.backpropagation.WeightUpdates;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

@RunWith(Parameterized.class)
public class BPCalculationProvidersComparisonTest extends AbstractTest
{
	private static final Logger logger = LoggerFactory.getLogger(BPCalculationProvidersComparisonTest.class);

	private TestConfiguration conf;

	public BPCalculationProvidersComparisonTest(TestConfiguration conf)
	{
		this.conf = conf;
	}

	@Parameters
	public static Collection<TestConfiguration[]> runtimeConfigurations()
	{
		List<TestConfiguration[]> configurations = new ArrayList<>();

		// conf1 (reference)
		RuntimeConfiguration conf1 = new RuntimeConfiguration();
		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);

//		RuntimeConfiguration conf2 = new RuntimeConfiguration();
//		conf2.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//		conf2.setUseDataSharedMemory(false);
//		conf2.setUseWeightsSharedMemory(false);

		// conf2 (for tests)
		RuntimeConfiguration conf2 = new RuntimeConfiguration();
		conf2.setCalculationProvider(CalculationProvider.OPENCL);
		conf2.setUseDataSharedMemory(false);
		conf2.setUseWeightsSharedMemory(false);
		conf2.getOpenCLConfiguration().setAggregateOperations(true);
		conf2.getOpenCLConfiguration().setUseOptionsString(false);
		conf2.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
		conf2.getOpenCLConfiguration().setPushToDeviceBeforeOperation(false);
		conf2.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(false);
		conf2.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);

		// trainer1
		{
			NeuralNetworkBuilder builder = new NeuralNetworkBuilder();
			builder.setOverrideAddBiasTo(true);
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 20, 1, 1));
			builder.addLayerBuilder(new FullyConnectedLayerBuilder(20).setActivationType(ActivationType.ReLU));
			builder.addLayerBuilder(new FullyConnectedLayerBuilder(10).setActivationType(ActivationType.SoftMax));
			builder.setTrainingSet(new RandomInputProvider(3, 20, 10, new Random(123)));

			builder.setLearningRate(0.1f);
			builder.setMomentum(0.9f);
			builder.setL1weightDecay(0.01f);
			builder.setL2weightDecay(0.01f);
			builder.setEpochs(1);
			builder.setTrainingBatchSize(3);

			configurations.add(new TestConfiguration[] { new TestConfiguration(conf1, conf2, builder, 123) });
		}

		// trainer1
		{
			NeuralNetworkBuilder builder = new NeuralNetworkBuilder();
			builder.setOverrideAddBiasTo(true);
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 20, 1, 1));
			builder.addLayerBuilder(new FullyConnectedLayerBuilder(20).setActivationType(ActivationType.ReLU));
			builder.addLayerBuilder(new FullyConnectedLayerBuilder(10).setActivationType(ActivationType.SoftMax));
			builder.setTrainingSet(new RandomInputProvider(3, 20, 10, new Random(123)));
			
			builder.setLearningRate(0.1f);
			builder.setMomentum(0.9f);
			builder.setL1weightDecay(0.01f);
			builder.setL2weightDecay(0.01f);
			builder.setEpochs(1);
			builder.setTrainingBatchSize(3);
			
			configurations.add(new TestConfiguration[] { new TestConfiguration(conf1, conf2, builder, 123) });
		}
		
		return configurations;
	}


	@Test
	public void compare()
	{
		// obtain values with conf1;
		Environment.getInstance().setRuntimeConfiguration(conf.conf1);
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair1 = conf.builder.buildWithTrainer();
		BackPropagationTrainer<?> trainer1 = (BackPropagationTrainer<?>) neuralNetworkTrainerPair1.getRight();
		trainer1.setRandomInitializer(new NNRandomInitializer(new RandomInitializerImpl(new Random(conf.randomSeed), -0.01f, 0.01f)));

		trainer1.train();

		ValuesProvider conf1Activations = trainer1.getActivations();
		ValuesProvider conf1Backpropagation = trainer1.getBackpropagation();
		ValuesProvider conf1Weights = neuralNetworkTrainerPair1.getLeft().getProperties().getParameter(Constants.WEIGHTS_PROVIDER);

		// obtain values with conf2;
		Environment.getInstance().setRuntimeConfiguration(conf.conf2);
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair2 = conf.builder.buildWithTrainer();
		BackPropagationTrainer<?> trainer2 = (BackPropagationTrainer<?>) neuralNetworkTrainerPair2.getRight();
		trainer2.setRandomInitializer(new NNRandomInitializer(new RandomInitializerImpl(new Random(conf.randomSeed), -0.01f, 0.01f)));

		trainer2.train();

		ValuesProvider conf2Activations = trainer2.getActivations();
		ValuesProvider conf2Backpropagation = trainer2.getBackpropagation();
		ValuesProvider conf2Weights = neuralNetworkTrainerPair2.getLeft().getProperties().getParameter(Constants.WEIGHTS_PROVIDER);

		// compare activations
		Iterator<Tensor> ita1 = conf1Activations.getTensors().iterator();
		Iterator<Tensor> ita2 = conf2Activations.getTensors().iterator();
		while (ita1.hasNext() && ita2.hasNext())
		{
			Tensor t1 = ita1.next();
			Tensor t2 = ita2.next();
			TensorIterator t1it = t1.iterator();
			TensorIterator t2it = t2.iterator();

			while (t1it.hasNext() && t2it.hasNext())
			{
				int t1index = t1it.next();
				int t2index = t2it.next();
				assertTrue(t1.getElements() != t2.getElements());
				assertTrue(t1.getElements()[t1index] != Float.NaN);
				assertTrue(t2.getElements()[t2index] != Float.NaN);
				assertTrue((t1.getElements()[t1index] != 0 && t2.getElements()[t2index] != 0) || (t1.getElements()[t1index] == 0 && t2.getElements()[t2index] == 0));

				try
				{
					assertEquals(0, Math.abs(t1.getElements()[t1index] - t2.getElements()[t2index]), 0.000001f);
				} catch (AssertionError ex)
				{
					String message = "ACT ";
					Object key = conf2Activations.getKey(t2);
					if (key instanceof Layer)
					{
						message += ((Layer) key).getName();
						LayerCalculatorImpl lc = (LayerCalculatorImpl) trainer2.getNeuralNetwork().getLayerCalculator();
						if (lc.getConnectionCalculator((Layer) key) instanceof ConnectionCalculatorImpl)
						{
							ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) lc.getConnectionCalculator((Layer) key);
							List<String> functions = new ArrayList<>();
							for (TensorFunction f : cc.getInputModifierFunctions())
							{
								functions.add(f.getClass().getSimpleName());
							}

							for (ConnectionCalculator c : cc.getInputFunctions())
							{
								functions.add(c.getClass().getSimpleName());
							}

							for (TensorFunction f : cc.getActivationFunctions())
							{
								functions.add(f.getClass().getSimpleName());
							}

							message += " " + functions.stream().collect(Collectors.joining("->"));
						}
					} else
					{
						message += key.getClass().getSimpleName();
					}

					message += Arrays.toString(t1it.getCurrentPosition()) + "; TARGET->ACTUAL: " + t1.getElements()[t1index] + "->" + t2.getElements()[t2index];

					logger.error(message);

					if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL)
					{
						for (OpenCLArrayReference ref : OpenCLArrayReferenceManager.getInstance().getArrayReferences(t2))
						{
							OpenCLCore.getInstance().checkFloatBuf(ref.getId(), t1.getElements());
						}
					}

					throw ex;
				}
			}
		}

		// compare bp activations
		Iterator<Tensor> itbp1 = conf1Backpropagation.getTensors().iterator();
		Iterator<Tensor> itbp2 = conf2Backpropagation.getTensors().iterator();

		while (itbp1.hasNext() && itbp2.hasNext())
		{
			Tensor t1 = itbp1.next();
			Tensor t2 = itbp2.next();
			TensorIterator t1it = t1.iterator();
			TensorIterator t2it = t2.iterator();

			while (t1it.hasNext() && t2it.hasNext())
			{
				int t1index = t1it.next();
				int t2index = t2it.next();
				assertTrue(t1.getElements() != t2.getElements());
				assertTrue(t1.getElements()[t1index] != Float.NaN);
				assertTrue(t2.getElements()[t2index] != Float.NaN);
				assertTrue((t1.getElements()[t1index] != 0 && t2.getElements()[t2index] != 0) || (t1.getElements()[t1index] == 0 && t2.getElements()[t2index] == 0));

				try
				{
					assertEquals(0, Math.abs(t1.getElements()[t1index] - t2.getElements()[t2index]), 0.000001f);
				} catch (AssertionError ex)
				{
					String message = "BP ";
					Object key = conf2Backpropagation.getKey(t2);

					if (key instanceof Layer)
					{
						message += ((Layer) key).getName();
						BackPropagationLayerCalculatorImpl lc = (BackPropagationLayerCalculatorImpl) trainer2.getBPLayerCalculator();
						if (lc.getConnectionCalculator((Layer) key) instanceof BackPropagationConnectionCalculatorImpl)
						{
							BackPropagationConnectionCalculatorImpl cc = (BackPropagationConnectionCalculatorImpl) lc.getConnectionCalculator((Layer) key);
							List<String> functions = new ArrayList<>();
							for (TensorFunction f : cc.getInputModifierFunctions())
							{
								functions.add(f.getClass().getSimpleName());
							}

							for (ConnectionCalculator c : cc.getInputFunctions())
							{
								functions.add(c.getClass().getSimpleName());
							}

							for (TensorFunction f : cc.getActivationFunctions())
							{
								functions.add(f.getClass().getSimpleName());
							}

							message += " " + functions.stream().collect(Collectors.joining("->"));
						}
					} else
					{
						message += key.getClass().getSimpleName();
					}

					message += Arrays.toString(t1it.getCurrentPosition()) + "; TARGET->ACTUAL: " + t1.getElements()[t1index] + "->" + t2.getElements()[t2index];

					logger.error(message);

					if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL)
					{
						for (OpenCLArrayReference ref : OpenCLArrayReferenceManager.getInstance().getArrayReferences(t2))
						{
							OpenCLCore.getInstance().checkFloatBuf(ref.getId(), t1.getElements());
						}
					}

					throw ex;
				}
			}
		}

		// compare weights
		Iterator<Tensor> itw1 = conf1Weights.getTensors().iterator();
		Iterator<Tensor> itw2 = conf2Weights.getTensors().iterator();

		while (itw1.hasNext() && itw2.hasNext())
		{
			Tensor t1 = itw1.next();
			Tensor t2 = itw2.next();

			TensorIterator t1it = t1.iterator();
			TensorIterator t2it = t2.iterator();
			while (t1it.hasNext() && t2it.hasNext())
			{
				int t1index = t1it.next();
				int t2index = t2it.next();
				assertTrue(t1.getElements() != t2.getElements());
				assertTrue(t1.getElements()[t1index] != Float.NaN);
				assertTrue(t2.getElements()[t2index] != Float.NaN);
				assertTrue((t1.getElements()[t1index] != 0 && t2.getElements()[t2index] != 0) || (t1.getElements()[t1index] == 0 && t2.getElements()[t2index] == 0));
				try
				{
					assertEquals(0, Math.abs(t1.getElements()[t1index] - t2.getElements()[t2index]), 0.000001f);
				} catch (AssertionError ex)
				{
					Connections c = (Connections) conf2Weights.getKey(t2);

					WeightUpdates wu = trainer2.getWeightUpdates().get(c);

					String message = "WU " + c.getInputLayer().getName() + "->" + c.getOutputLayer().getName() + " " + wu.getClass().getSimpleName() + " " + Arrays.toString(t1it.getCurrentPosition()) + "; TARGET->ACTUAL: " + t1.getElements()[t1index]
							+ "->" + t2.getElements()[t2index];

					logger.error(message);

					if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL)
					{
						for (OpenCLArrayReference ref : OpenCLArrayReferenceManager.getInstance().getArrayReferences(t2))
						{
							OpenCLCore.getInstance().checkFloatBuf(ref.getId(), t1.getElements());
						}
					}

					throw ex;
				}
			}
		}
	}

	private static class TestConfiguration implements Serializable
	{
		private static final long serialVersionUID = 1L;

		private RuntimeConfiguration conf1;
		private RuntimeConfiguration conf2;
		private NeuralNetworkBuilder builder;
		private long randomSeed;

		public TestConfiguration(RuntimeConfiguration conf1, RuntimeConfiguration conf2, NeuralNetworkBuilder builder, long randomSeed)
		{
			super();
			this.conf1 = conf1;
			this.conf2 = conf2;
			this.builder = builder;
			this.randomSeed = randomSeed;
		}
	}
}
