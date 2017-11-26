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
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.layer.ConvolutionalLayerBuilder;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.PoolingLayerBuilder;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.random.RandomInitializerImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.RuntimeConfiguration;

@RunWith(Parameterized.class)
@Ignore
@Deprecated /// test moved to core
public class CNNStepByStepTest
{
	public CNNStepByStepTest(RuntimeConfiguration conf)
	{
		Environment.getInstance().setRuntimeConfiguration(conf);
	}

	@Parameters
	public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
	{
		NeuralNetworkBuilder builder = new NeuralNetworkBuilder();

		Random r = new Random(123456789);
		// network
		{
			builder.addLayerBuilder(new InputLayerBuilder("inputLayer", 32, 32, 3));

			// first part
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 16);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, -0.01f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasL2weightDecay(0.0001f);
				convolutionalLayerBuilder.setActivationType(ActivationType.ReLU);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// second part
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 20);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, -0.01f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasL2weightDecay(0.0001f);
				convolutionalLayerBuilder.setActivationType(ActivationType.ReLU);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// third part
			{
				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(5, 20);
				convolutionalLayerBuilder.setPaddingSize(2);
				convolutionalLayerBuilder.setStrideSize(1);
				convolutionalLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, -0.01f, 0.01f));
				convolutionalLayerBuilder.setLearningRate(0.01f);
				convolutionalLayerBuilder.setMomentum(0.9f);
				convolutionalLayerBuilder.setBiasLearningRate(0.01f);
				convolutionalLayerBuilder.setBiasL2weightDecay(0.0001f);
				convolutionalLayerBuilder.setActivationType(ActivationType.ReLU);
				builder.addLayerBuilder(convolutionalLayerBuilder);

				PoolingLayerBuilder poolingLayerBuilder = new PoolingLayerBuilder(2);
				poolingLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
				poolingLayerBuilder.setStrideSize(2);
				builder.addLayerBuilder(poolingLayerBuilder);
			}

			// fully connected layer
			{
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(10);
				fullyConnectedLayerBuilder.setWeightInitializer(new RandomInitializerImpl(r, -0.01f, 0.01f));
				fullyConnectedLayerBuilder.setLearningRate(0.01f);
				fullyConnectedLayerBuilder.setMomentum(0.9f);
				fullyConnectedLayerBuilder.setBiasLearningRate(0.01f);
				fullyConnectedLayerBuilder.setBiasL2weightDecay(0.0001f);
				fullyConnectedLayerBuilder.setActivationType(ActivationType.SoftMax);
				builder.addLayerBuilder(fullyConnectedLayerBuilder);
			}
		}

		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> pair = builder.buildWithTrainer();

		List<RuntimeConfiguration[]> configurations = new ArrayList<>();

		TrainerRuntimeConfiguration conf1 = new TrainerRuntimeConfiguration();
		conf1.trainer = (BackPropagationTrainer<NeuralNetwork>) pair.getRight();
		conf1.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);
		configurations.add(new RuntimeConfiguration[] { conf1 });

//			TrainerRuntimeConfiguration conf3 = new TrainerRuntimeConfiguration();
//			conf3.trainer = (BackPropagationTrainer<NeuralNetwork>) pair.getRight();
//			conf3.setCalculationProvider(CalculationProvider.OPENCL);
//			conf3.setUseDataSharedMemory(false);
//			conf3.setUseWeightsSharedMemory(false);
//			conf3.getOpenCLConfiguration().setAggregateOperations(false);
//			conf3.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
//			conf3.getOpenCLConfiguration().setPushToDeviceBeforeOperation(true);
//			conf3.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(false);
//			conf3.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//			configurations.add(new RuntimeConfiguration[] { conf3 });

		return configurations;
	}

	@Test
	public void testConstruction()
	{
		TrainerRuntimeConfiguration rt = (TrainerRuntimeConfiguration) Environment.getInstance().getRuntimeConfiguration();
		NeuralNetworkImpl nn = (NeuralNetworkImpl) rt.trainer.getNeuralNetwork();
		Layer input = nn.getInputLayer();

		assertEquals(1, input.getConnections().size());

		Conv2DConnection conv1c = (Conv2DConnection) input.getConnections().get(0);
		assertEquals(3, conv1c.getInputFilters());
		assertEquals(16, conv1c.getOutputFilters());
		assertEquals(5, conv1c.getFilterRows());
		assertEquals(5, conv1c.getFilterColumns());
		assertEquals(32, conv1c.getInputFeatureMapRows());
		assertEquals(32, conv1c.getInputFeatureMapColumns());
		assertEquals(1, conv1c.getRowStride());
		assertEquals(1, conv1c.getColumnStride());
		assertEquals(2, conv1c.getOutputColumnPadding());
		assertEquals(2, conv1c.getOutputRowPadding());
		assertEquals(32, conv1c.getOutputFeatureMapRowsWithPadding());
		assertEquals(32, conv1c.getOutputFeatureMapColumnsWithPadding());
		assertEquals(28 * 28 * 16, conv1c.getOutputUnitCount());

		Conv2DConnection conv1b = (Conv2DConnection) conv1c.getOutputLayer().getConnections().get(1);
		assertEquals(1, conv1b.getInputFilters());
		assertEquals(16, conv1b.getOutputFilters());
		assertEquals(1, conv1b.getFilterRows());
		assertEquals(1, conv1b.getFilterColumns());
		assertEquals(28, conv1b.getInputFeatureMapRows());
		assertEquals(28, conv1b.getInputFeatureMapColumns());
		assertEquals(1, conv1b.getRowStride());
		assertEquals(1, conv1b.getColumnStride());
		assertEquals(2, conv1b.getOutputColumnPadding());
		assertEquals(2, conv1b.getOutputRowPadding());
		assertEquals(28 * 28 * 16, conv1b.getOutputUnitCount());
		assertEquals(32 * 32 * 16, conv1b.getOutputUnitCountWithPadding());

		Subsampling2DConnection pool1 = (Subsampling2DConnection) conv1c.getOutputLayer().getConnections().get(2);
		assertEquals(16, pool1.getFilters());
		assertEquals(2, pool1.getColumnStride());
		assertEquals(2, pool1.getRowStride());
		assertEquals(16, pool1.getOutputFeatureMapRows());
		assertEquals(16, pool1.getOutputFeatureMapColumns());
		assertEquals(16, pool1.getOutputFeatureMapRowsWithPadding());
		assertEquals(16, pool1.getOutputFeatureMapColumnsWithPadding());
		assertEquals(32 * 32 * 16, pool1.getInputUnitCount());
		assertEquals(16 * 16 * 16, pool1.getOutputUnitCount());

		Conv2DConnection conv2c = (Conv2DConnection) pool1.getOutputLayer().getConnections().get(1);
		assertEquals(16, conv2c.getInputFilters());
		assertEquals(20, conv2c.getOutputFilters());
		assertEquals(5, conv2c.getFilterRows());
		assertEquals(5, conv2c.getFilterColumns());
		assertEquals(16, conv2c.getInputFeatureMapRows());
		assertEquals(16, conv2c.getInputFeatureMapColumns());
		assertEquals(1, conv2c.getRowStride());
		assertEquals(1, conv2c.getColumnStride());
		assertEquals(2, conv2c.getOutputColumnPadding());
		assertEquals(2, conv2c.getOutputRowPadding());
		assertEquals(12, conv2c.getOutputFeatureMapRows());
		assertEquals(12, conv2c.getOutputFeatureMapColumns());
		assertEquals(16, conv2c.getOutputFeatureMapRowsWithPadding());
		assertEquals(16, conv2c.getOutputFeatureMapColumnsWithPadding());
		assertEquals(12 * 12 * 20, conv2c.getOutputUnitCount());
		assertEquals(16 * 16 * 20, conv2c.getOutputUnitCountWithPadding());
		
		Conv2DConnection conv2b = (Conv2DConnection) conv2c.getOutputLayer().getConnections().get(1);
		assertEquals(1, conv2b.getInputFilters());
		assertEquals(20, conv2b.getOutputFilters());
		assertEquals(1, conv2b.getFilterRows());
		assertEquals(1, conv2b.getFilterColumns());
		assertEquals(12, conv2b.getInputFeatureMapRows());
		assertEquals(12, conv2b.getInputFeatureMapColumns());
		assertEquals(1, conv2b.getRowStride());
		assertEquals(1, conv2b.getColumnStride());
		assertEquals(2, conv2b.getOutputColumnPadding());
		assertEquals(2, conv2b.getOutputRowPadding());
		assertEquals(12, conv2b.getOutputFeatureMapRows());
		assertEquals(12, conv2b.getOutputFeatureMapColumns());
		assertEquals(16, conv2b.getOutputFeatureMapRowsWithPadding());
		assertEquals(16, conv2b.getOutputFeatureMapColumnsWithPadding());
		assertEquals(12 * 12 * 20, conv2b.getOutputUnitCount());
		assertEquals(16 * 16 * 20, conv2b.getOutputUnitCountWithPadding());

		Subsampling2DConnection pool2 = (Subsampling2DConnection) conv2c.getOutputLayer().getConnections().get(2);
		assertEquals(20, pool2.getFilters());
		assertEquals(2, pool2.getColumnStride());
		assertEquals(2, pool2.getRowStride());
		assertEquals(8, pool2.getOutputFeatureMapRows());
		assertEquals(8, pool2.getOutputFeatureMapColumns());
		assertEquals(8, pool2.getOutputFeatureMapRowsWithPadding());
		assertEquals(8, pool2.getOutputFeatureMapColumnsWithPadding());
		assertEquals(16 * 16 * 20, pool2.getInputUnitCount());
		assertEquals(8 * 8 * 20, pool2.getOutputUnitCount());

		Conv2DConnection conv3c = (Conv2DConnection) pool2.getOutputLayer().getConnections().get(1);
		assertEquals(20, conv3c.getInputFilters());
		assertEquals(20, conv3c.getOutputFilters());
		assertEquals(5, conv3c.getFilterRows());
		assertEquals(5, conv3c.getFilterColumns());
		assertEquals(8, conv3c.getInputFeatureMapRows());
		assertEquals(8, conv3c.getInputFeatureMapColumns());
		assertEquals(1, conv3c.getRowStride());
		assertEquals(1, conv3c.getColumnStride());
		assertEquals(2, conv3c.getOutputColumnPadding());
		assertEquals(2, conv3c.getOutputRowPadding());
		assertEquals(4, conv3c.getOutputFeatureMapRows());
		assertEquals(4, conv3c.getOutputFeatureMapColumns());
		assertEquals(8, conv3c.getOutputFeatureMapRowsWithPadding());
		assertEquals(8, conv3c.getOutputFeatureMapColumnsWithPadding());
		assertEquals(4 * 4 * 20, conv3c.getOutputUnitCount());
		assertEquals(8 * 8 * 20, conv3c.getOutputUnitCountWithPadding());

		Conv2DConnection conv3b = (Conv2DConnection) conv3c.getOutputLayer().getConnections().get(1);
		assertEquals(1, conv3b.getInputFilters());
		assertEquals(20, conv3b.getOutputFilters());
		assertEquals(1, conv3b.getFilterRows());
		assertEquals(1, conv3b.getFilterColumns());
		assertEquals(4, conv3b.getInputFeatureMapRows());
		assertEquals(4, conv3b.getInputFeatureMapColumns());
		assertEquals(1, conv3b.getRowStride());
		assertEquals(1, conv3b.getColumnStride());
		assertEquals(2, conv3b.getOutputColumnPadding());
		assertEquals(2, conv3b.getOutputRowPadding());
		assertEquals(4, conv3b.getOutputFeatureMapRows());
		assertEquals(4, conv3b.getOutputFeatureMapColumns());
		assertEquals(8, conv3b.getOutputFeatureMapRowsWithPadding());
		assertEquals(8, conv3b.getOutputFeatureMapColumnsWithPadding());
		assertEquals(4 * 4 * 20, conv3b.getOutputUnitCount());
		assertEquals(8 * 8 * 20, conv3b.getOutputUnitCountWithPadding());

		Subsampling2DConnection pool3 = (Subsampling2DConnection) conv3c.getOutputLayer().getConnections().get(2);
		assertEquals(20, pool3.getFilters());
		assertEquals(2, pool3.getColumnStride());
		assertEquals(2, pool3.getRowStride());
		assertEquals(4, pool3.getOutputFeatureMapRows());
		assertEquals(4, pool3.getOutputFeatureMapColumns());
		assertEquals(4, pool3.getOutputFeatureMapRowsWithPadding());
		assertEquals(4, pool3.getOutputFeatureMapColumnsWithPadding());
		assertEquals(8 * 8 * 20, pool3.getInputUnitCount());
		assertEquals(4 * 4 * 20, pool3.getOutputUnitCount());

		FullyConnected fc1 = (FullyConnected) pool3.getOutputLayer().getConnections().get(1);
		assertEquals(4 * 4 * 20, fc1.getInputUnitCount());
		assertEquals(10, fc1.getOutputUnitCount());
	}

	private static class TrainerRuntimeConfiguration extends RuntimeConfiguration
	{
		private static final long serialVersionUID = 1L;

		public BackPropagationTrainer<NeuralNetwork> trainer;
	}
}
