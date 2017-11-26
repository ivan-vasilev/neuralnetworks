package com.github.neuralnetworks.samples.test;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.layer.ConvolutionalLayerBuilder;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.PoolingLayerBuilder;
import com.github.neuralnetworks.input.SimpleFileInputProvider;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.events.TrainerSaveListener;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;
import com.github.neuralnetworks.util.Serializer;

/**
 * 
 */
@Ignore
@Deprecated
@RunWith(Parameterized.class)
public class TestSerialization
{
	public TestSerialization(RuntimeConfiguration conf)
	{
		Environment.getInstance().setRuntimeConfiguration(conf);
	}

	@Parameters
	public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
	{
		List<RuntimeConfiguration[]> configurations = new ArrayList<>();

		RuntimeConfiguration conf1 = new RuntimeConfiguration();
		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);
		configurations.add(new RuntimeConfiguration[] { conf1 });

//		RuntimeConfiguration conf3 = new RuntimeConfiguration();
//		conf3.setCalculationProvider(CalculationProvider.OPENCL);
//		conf3.setUseDataSharedMemory(false);
//		conf3.setUseWeightsSharedMemory(false);
//		conf3.getOpenCLConfiguration().setAggregateOperations(true);
//		conf3.getOpenCLConfiguration().setPreferredDevice(2);
//		conf3.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);
//		conf3.getOpenCLConfiguration().setPushToDeviceBeforeOperation(false);
//		conf3.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(true);
//		conf3.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//		configurations.add(new RuntimeConfiguration[] { conf3 });

		return configurations;
	}

	@Test
	public void test8()
	{
		boolean readFromFile = true;
		BackPropagationTrainer<?> bpt = null;
		String filename = "FILE_PATH";
		if (readFromFile)
		{
			bpt = (BackPropagationTrainer<?>) Serializer.loadTrainer(filename);
			bpt.setEpochs(1);
		} else
		{
			NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

			Random r = new Random(123);

			// network
			{
				builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 28, 28, 1));

				// conv
				{
					ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 6);
					convolutionalLayerBuilder.setPaddingSize(0);
					convolutionalLayerBuilder.setStrideSize(1);
					convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
					convolutionalLayerBuilder.setLearningRate(0.07f);
					convolutionalLayerBuilder.setMomentum(0.9f);
					convolutionalLayerBuilder.setBiasLearningRate(0.1f);
					convolutionalLayerBuilder.setBiasMomentum(0.9f);
					convolutionalLayerBuilder.setActivationType(ActivationType.Nothing);
					builder.addLayerBuilder(convolutionalLayerBuilder);

					PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
					poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
					poolingLayerBuilder.setActivationType(ActivationType.Nothing);
					poolingLayerBuilder.setStrideSize(2);
					builder.addLayerBuilder(poolingLayerBuilder);
				}

				// conv
				{
					ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 14);
					convolutionalLayerBuilder.setPaddingSize(0);
					convolutionalLayerBuilder.setStrideSize(1);
					convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0f, 0.01f));
					convolutionalLayerBuilder.setLearningRate(0.07f);
					convolutionalLayerBuilder.setMomentum(0.9f);
					convolutionalLayerBuilder.setBiasLearningRate(0.1f);
					convolutionalLayerBuilder.setBiasMomentum(0.9f);
					convolutionalLayerBuilder.setActivationType(ActivationType.Nothing);
					builder.addLayerBuilder(convolutionalLayerBuilder);

					PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
					poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
					poolingLayerBuilder.setActivationType(ActivationType.ReLU);
					poolingLayerBuilder.setStrideSize(2);
					builder.addLayerBuilder(poolingLayerBuilder);
				}

				// fc
				{
					FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(140);
					fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0, 0.01f));
					fullyConnectedLayerBuilder.setLearningRate(0.1f);
					fullyConnectedLayerBuilder.setMomentum(0.9f);
					fullyConnectedLayerBuilder.setBiasLearningRate(0.1f);
					fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
					fullyConnectedLayerBuilder.setActivationType(ActivationType.ReLU);
					builder.addLayerBuilder(fullyConnectedLayerBuilder);
				}

				// fc
				{
					FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(70);
					fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0, 0.01f));
					fullyConnectedLayerBuilder.setLearningRate(0.1f);
					fullyConnectedLayerBuilder.setMomentum(0.9f);
					fullyConnectedLayerBuilder.setBiasLearningRate(0.1f);
					fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
					fullyConnectedLayerBuilder.setActivationType(ActivationType.ReLU);
					builder.addLayerBuilder(fullyConnectedLayerBuilder);
				}

				// fc
				{
					FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(10);
					fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, 0, 0.01f));
					fullyConnectedLayerBuilder.setLearningRate(0.1f);
					fullyConnectedLayerBuilder.setMomentum(0.9f);
					fullyConnectedLayerBuilder.setBiasLearningRate(0.1f);
					fullyConnectedLayerBuilder.setBiasMomentum(0.9f);
					fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
					builder.addLayerBuilder(fullyConnectedLayerBuilder);
				}

				// trainer
				{
					TrainingInputProvider trainInputProvider = new SimpleFileInputProvider("mnist/train-images.float", "mnist/train-labels.float", 28 * 28, 10, 60000);
					TrainingInputProvider testInputProvider = new SimpleFileInputProvider("mnist/t10k-images.float", "mnist/t10k-labels.float", 28 * 28, 10, 10000);
					builder.setTrainingSet(trainInputProvider);
					builder.setTestingSet(testInputProvider);

					builder.setEpochs(3);
					builder.setTrainingBatchSize(100);
					builder.setTestBatchSize(1000);
				}

				bpt = (BackPropagationTrainer<?>) builder.buildWithTrainer().getRight();

				LogTrainingListener ls = new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true);
				ls.setLogBatchLoss(true);
				bpt.addEventListener(ls);

				// save the listener each epochs
				bpt.addEventListener(new TrainerSaveListener(filename, false));
			}
		}
		
		// training
		bpt.train();

		// test
		bpt.test();
		
		assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
	}
}
