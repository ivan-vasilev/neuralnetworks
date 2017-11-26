package com.github.neuralnetworks.builder.designio.protobuf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.layer.ConvolutionalLayerBuilder;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.PoolingLayerBuilder;
import com.github.neuralnetworks.builder.layer.structure.BiasLayerConnectable;
import com.github.neuralnetworks.builder.layer.structure.DropOutableLayer;
import com.github.neuralnetworks.builder.layer.structure.LayerBuilder;
import com.github.neuralnetworks.builder.layer.structure.LearnableLayer;
import com.github.neuralnetworks.builder.layer.structure.MainFunctionsChangeable;
import com.github.neuralnetworks.training.random.DummyFixedInitializer;
import com.github.neuralnetworks.training.random.NormalDistributionInitializer;
import com.github.neuralnetworks.training.random.RandomInitializer;

/**
 * @author tmey
 */
public class ProtoBufNNConfigMapper
{

	public static NeuralNetworkBuilder mapProtoBufTo(ProtoBufWrapper.NetParameter netParameter, ProtoBufWrapper.SolverParameter solverParameter)
	{
		NeuralNetworkBuilder neuralNetworkBuilder = new NeuralNetworkBuilder();

		mapProtoBufTrainConfigToBuilder(neuralNetworkBuilder, solverParameter);
		mapProtoBufNetConfigToBuilder(neuralNetworkBuilder, netParameter);

		return neuralNetworkBuilder;
	}

	public static NeuralNetworkBuilder mapProtoBufNetConfigToBuilder(ProtoBufWrapper.NetParameter netParameter)
	{
		return mapProtoBufNetConfigToBuilder(new NeuralNetworkBuilder(), netParameter);
	}

	/**
	 * map neural network configuration to the builder
	 * 
	 * @param neuralNetworkBuilder
	 * @param netParameter
	 * @return
	 */
	public static NeuralNetworkBuilder mapProtoBufNetConfigToBuilder(NeuralNetworkBuilder neuralNetworkBuilder, ProtoBufWrapper.NetParameter netParameter)
	{
		if (neuralNetworkBuilder == null)
		{
			throw new IllegalArgumentException("neuralNetworkBuilder must be not null!");
		}

		if (netParameter == null)
		{
			throw new IllegalArgumentException("netParameter must be not null!");
		}

		// create input layer
		if (netParameter.getInputDimCount() != 4)
		{
			throw new IllegalArgumentException("the number of inputdim must 4! (it is " + (netParameter.getInputDimCount()) + ")");
		}

		if (netParameter.getInputCount() != 1)
		{
			throw new IllegalArgumentException("more than one input isn't supported but there must be at least one! (" + netParameter.getInputCount() + ")");
		}

		String inputLayerName = netParameter.getInput(0);

		InputLayerBuilder inputLayerBuilder = new InputLayerBuilder(inputLayerName, netParameter.getInputDim(2), netParameter.getInputDim(3), netParameter.getInputDim(1));
		neuralNetworkBuilder.addLayerBuilder(inputLayerBuilder);


		// create infrastructure
		Map<String, ProtoBufWrapper.LayerParameter> mapNameToLayerParameter = new HashMap<>();
		Map<String, List<ProtoBufWrapper.LayerParameter>> mapBottomToLayerFunctions = new HashMap<>();
		Map<String, ProtoBufWrapper.LayerParameter> mapBottomToLayerParameter = new HashMap<>();


		ProtoBufWrapper.LayerParameter currentLayer = null;

		for (ProtoBufWrapper.LayerParameter layerParameter : netParameter.getLayersList())
		{
			if (!layerParameter.hasName())
			{
				throw new IllegalArgumentException("Each layer needs a name!");
			}

			if (!layerParameter.hasType())
			{
				throw new IllegalArgumentException("each layer need a type! - " + layerParameter.getName());
			}

			// if data layer
			if (layerParameter.getType() == ProtoBufWrapper.LayerParameter.LayerType.DATA)
			{
				if (layerParameter.hasDataParam())
				{
					// check if train was uses if something was uses
					if (layerParameter.getIncludeCount() > 0)
					{
						boolean hasTrain = false;
						for (int i = 0; i < layerParameter.getIncludeCount(); i++)
						{
							if (layerParameter.getInclude(i).getPhase() == ProtoBufWrapper.Phase.TRAIN)
							{
								hasTrain = true;
								break;
							}
						}
						if (!hasTrain)
						{
							continue;
						}
					}
					// get batch size
					int batchSize = layerParameter.getDataParam().getBatchSize();
					neuralNetworkBuilder.setTrainingBatchSize(batchSize);
					neuralNetworkBuilder.setTestBatchSize(batchSize);
				}
				continue;
			}

			if (layerParameter.getBottomCount() != 1)
			{
				throw new IllegalArgumentException("Only exact one input layer (bottom) accepted! - " + layerParameter.getName());
			}

			if (layerParameter.getTopCount() != 1)
			{
				throw new IllegalArgumentException("Only exact one output layer (top) accepted! - " + layerParameter.getName());
			}

			// add to correct map
			ProtoBufWrapper.LayerParameter.LayerType layerType = layerParameter.getType();

			if (layerType == ProtoBufWrapper.LayerParameter.LayerType.CONVOLUTION
					|| layerType == ProtoBufWrapper.LayerParameter.LayerType.INNER_PRODUCT
					|| layerType == ProtoBufWrapper.LayerParameter.LayerType.POOLING)
			{
				if (mapNameToLayerParameter.put(layerParameter.getName(), layerParameter) != null)
				{
					throw new IllegalArgumentException("Each layer name must be unique! Please check " + layerParameter.getName() + "!");
				}

				if (mapBottomToLayerParameter.put(layerParameter.getBottom(0), layerParameter) != null)
				{
					throw new IllegalArgumentException("Two layers can't use the same input at the moment! Please check layers with the input" + layerParameter.getBottom(0) + "!");
				}
			} else if (layerType == ProtoBufWrapper.LayerParameter.LayerType.RELU
					|| layerType == ProtoBufWrapper.LayerParameter.LayerType.DROPOUT
					|| layerType == ProtoBufWrapper.LayerParameter.LayerType.SOFTMAX
					|| layerType == ProtoBufWrapper.LayerParameter.LayerType.SIGMOID)
			{
				if (!mapBottomToLayerFunctions.containsKey(layerParameter.getBottom(0)))
				{
					mapBottomToLayerFunctions.put(layerParameter.getBottom(0), new ArrayList<>());
				}

				mapBottomToLayerFunctions.get(layerParameter.getBottom(0)).add(layerParameter);

			} else
			{
				throw new IllegalArgumentException("Unknown layer type: " + layerType + "!");
			}

			// check if first layer that uses input

			if (layerParameter.getBottom(0).equals(inputLayerName))
			{
				if (currentLayer != null)
				{
					throw new IllegalArgumentException("There exists minimum two layer that uses the input! This is not supported at the moment! " + currentLayer.getName() + " and " + layerParameter.getName());
				}
				currentLayer = layerParameter;
			}

		}

		if (currentLayer == null)
		{
			throw new IllegalArgumentException("The input isn't used from any layer!");
		}


		// parse layer configuration
		while (currentLayer != null)
		{

			String layerName = currentLayer.getName();

			LayerBuilder layerBuilder;

			switch (currentLayer.getType())
			{
			case CONVOLUTION:
			{
				if (!currentLayer.hasConvolutionParam())
				{
					throw new IllegalArgumentException("The parameter for the convolutional layer (" + layerName + ") are missing!");
				}
				ProtoBufWrapper.ConvolutionParameter convolutionParam = currentLayer.getConvolutionParam();

				if (!convolutionParam.hasKernelSize())
				{
					throw new IllegalArgumentException("kernel size is missing in " + layerName + "!");
				}

				if (!convolutionParam.hasNumOutput())
				{
					throw new IllegalArgumentException("number of kernels is missing " + layerName + "! (NumOutput)");
				}

				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(convolutionParam.getKernelSize(), convolutionParam.getNumOutput());

				convolutionalLayerBuilder.setInputLayerName(currentLayer.getBottom(0));
				convolutionalLayerBuilder.setName(currentLayer.getName());

				if (convolutionParam.hasGroup() && convolutionParam.getGroup() > 1)
				{
					throw new IllegalArgumentException("A group larger than 1 isn't supported at the moment! - " + layerName);
				}

				if (convolutionParam.hasStride())
				{
					convolutionalLayerBuilder.setStrideSize(convolutionParam.getStride());
				}

				if (convolutionParam.hasPad())
				{
					convolutionalLayerBuilder.setPaddingSize(convolutionParam.getPad());
				}

				boolean useBias = true;

				if (convolutionParam.hasBiasTerm())
				{
					convolutionalLayerBuilder.setAddBias(convolutionParam.getBiasTerm());
					useBias = convolutionParam.getBiasTerm();
				}

				// initialize learning rates
				parseLearningParameter(convolutionalLayerBuilder, currentLayer, neuralNetworkBuilder.getLearningRate(), neuralNetworkBuilder.getL1weightDecay(), useBias);

				// weight initializer

				if (convolutionParam.hasWeightFiller())
				{
					convolutionalLayerBuilder.setWeightInitializer(getWeightIninitializer(convolutionParam.getWeightFiller()));
				}

				if (convolutionParam.hasBiasFiller())
				{
					convolutionalLayerBuilder.setBiasWeightInitializer(getWeightIninitializer(convolutionParam.getBiasFiller()));
				}

				layerBuilder = convolutionalLayerBuilder;
				break;
			}
			case POOLING:
			{
				if (!currentLayer.hasPoolingParam())
				{
					throw new IllegalArgumentException("The parameter for the pooling layer " + layerName + " are missing!");
				}
				ProtoBufWrapper.PoolingParameter poolingParam = currentLayer.getPoolingParam();

				if (!poolingParam.hasKernelSize())
				{
					throw new IllegalArgumentException("kernel size is missing in " + layerName + "!");
				}

				PoolingLayerBuilder poolingLayerBuilderLayerBuilder = new PoolingLayerBuilder(poolingParam.getKernelSize());

				poolingLayerBuilderLayerBuilder.setInputLayerName(currentLayer.getBottom(0));
				poolingLayerBuilderLayerBuilder.setName(currentLayer.getName());

				if (poolingParam.hasStride())
				{
					poolingLayerBuilderLayerBuilder.setStrideSize(poolingParam.getStride());
				}

				if (poolingParam.hasPool())
				{

					switch (poolingParam.getPool())
					{
					case MAX:
						poolingLayerBuilderLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
						break;
					case AVE:
						poolingLayerBuilderLayerBuilder.setTransferFunctionType(TransferFunctionType.Average_Pooling2D);
						break;
					default:
						throw new IllegalArgumentException("Unknown pooling type in " + layerName + "! " + poolingParam.getPool());
					}

				}
				layerBuilder = poolingLayerBuilderLayerBuilder;
				break;

			}
			case INNER_PRODUCT:
			{
				if (!currentLayer.hasInnerProductParam())
				{
					throw new IllegalArgumentException("The parameter for the fully connected layer " + layerName + " are missing!");
				}
				ProtoBufWrapper.InnerProductParameter innerProductParam = currentLayer.getInnerProductParam();

				if (!innerProductParam.hasNumOutput())
				{
					throw new IllegalArgumentException("The number of neurons (NumOutput) for the fully connected layer " + layerName + " are missing!");
				}

				// build layer
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(innerProductParam.getNumOutput());

				fullyConnectedLayerBuilder.setInputLayerName(currentLayer.getBottom(0));
				fullyConnectedLayerBuilder.setName(currentLayer.getName());

				boolean useBias = true;

				if (innerProductParam.hasBiasTerm())
				{
					fullyConnectedLayerBuilder.setAddBias(innerProductParam.getBiasTerm());
					useBias = innerProductParam.getBiasTerm();
				}

				// initialize learning rates
				parseLearningParameter(fullyConnectedLayerBuilder, currentLayer, neuralNetworkBuilder.getLearningRate(), neuralNetworkBuilder.getL1weightDecay(), useBias);

				// weight initialization

				if (innerProductParam.hasWeightFiller())
				{
					fullyConnectedLayerBuilder.setWeightInitializer(getWeightIninitializer(innerProductParam.getWeightFiller()));
				}

				if (innerProductParam.hasBiasFiller())
				{
					fullyConnectedLayerBuilder.setBiasWeightInitializer(getWeightIninitializer(innerProductParam.getBiasFiller()));
				}

				layerBuilder = fullyConnectedLayerBuilder;

				break;
			}
			default:
			{
				throw new IllegalArgumentException("Unsupported layer type " + currentLayer.getType() + " in " + layerName);
			}
			}

			// add functions and features

			if (mapBottomToLayerFunctions.containsKey(currentLayer.getName()))
			{
				for (ProtoBufWrapper.LayerParameter layerParameter : mapBottomToLayerFunctions.get(currentLayer.getName()))
				{
					switch (layerParameter.getType())
					{
					case DROPOUT:
						if (layerBuilder instanceof DropOutableLayer)
						{
							if (layerParameter.hasDropoutParam() && layerParameter.getDropoutParam().hasDropoutRatio())
							{
								((DropOutableLayer) layerBuilder).setDropoutRate(layerParameter.getDropoutParam().getDropoutRatio());
							}
							else
							{
								throw new IllegalArgumentException("A drop out layer must have a DropoutParam and a DropoutRatio! - " + currentLayer.getName() + "/" + layerParameter.getName());
							}
						}
						break;
					case RELU:
						if (layerBuilder instanceof MainFunctionsChangeable)
						{
							((MainFunctionsChangeable) layerBuilder).setActivationType(ActivationType.ReLU);
						}
						break;
					case SOFTMAX:
						if (layerBuilder instanceof MainFunctionsChangeable)
						{
							((MainFunctionsChangeable) layerBuilder).setActivationType(ActivationType.SoftMax);
						}
						break;
					case SIGMOID:
						if (layerBuilder instanceof MainFunctionsChangeable)
						{
							((MainFunctionsChangeable) layerBuilder).setActivationType(ActivationType.Sigmoid);
						}
						break;
					default:
						throw new IllegalArgumentException("Unknown layer type " + layerParameter.getType());

					}
				}

			}


			// add layer builder to network builder
			neuralNetworkBuilder.addLayerBuilder(layerBuilder);


			// get next layer
			if (!mapBottomToLayerParameter.containsKey(layerName))
			{
				currentLayer = null;
			} else
			{
				currentLayer = mapBottomToLayerParameter.get(layerName);
				mapBottomToLayerParameter.remove(layerName);
			}
		}

		return neuralNetworkBuilder;
	}

	private static RandomInitializer getWeightIninitializer(ProtoBufWrapper.FillerParameter weightFiller)
	{
		if (!weightFiller.hasType())
		{
			throw new IllegalArgumentException("Each weight filler needs a type!");
		}

		switch (weightFiller.getType())
		{
		case "constant":
		{
			float value = 0;
			if (weightFiller.hasValue())
			{
				value = weightFiller.getValue();
			}

			return new DummyFixedInitializer(value);
		}
		case "gaussian":
			float std;
			float mean = 0;
			if (!weightFiller.hasStd())
			{
				throw new IllegalArgumentException("A gaussian initializer needs the standard deviation (std)!");
			}
			std = weightFiller.getStd();

			if (weightFiller.hasMean())
			{
				mean = weightFiller.getMean();
			}

			return new NormalDistributionInitializer(mean, std);
		default:
			throw new IllegalArgumentException("Unknown weight filler type: " + weightFiller.getType());
		}
	}

	/**
	 * set the learning rate and weight decay for current layer and bias layer
	 * 
	 * @param layerBuilder
	 * @param layerParameter
	 */
	public static void parseLearningParameter(LayerBuilder layerBuilder, ProtoBufWrapper.LayerParameter layerParameter, float generalLearningRage, float generalL1WeightDecay, boolean useBias)
	{
		if (layerBuilder instanceof LearnableLayer)
		{
			if (layerParameter.getBlobsLrCount() > 0)
			{
				((LearnableLayer) layerBuilder).setLearningRate(generalLearningRage * layerParameter.getBlobsLr(0));
			}
			if (layerParameter.getWeightDecayCount() > 0)
			{
				((LearnableLayer) layerBuilder).setL1weightDecay(generalL1WeightDecay * layerParameter.getWeightDecay(0));
			}
		}

		if (layerBuilder instanceof BiasLayerConnectable && useBias)
		{
			if (layerParameter.getBlobsLrCount() > 1)
			{
				((BiasLayerConnectable) layerBuilder).setBiasLearningRate(generalLearningRage * layerParameter.getBlobsLr(1));
			}
			if (layerParameter.getWeightDecayCount() > 1)
			{
				((BiasLayerConnectable) layerBuilder).setBiasL1weightDecay(generalL1WeightDecay * layerParameter.getWeightDecay(1));
			}
		}
	}

	public static NeuralNetworkBuilder mapProtoBufTrainConfigToBuilder(ProtoBufWrapper.SolverParameter solverParameter)
	{
		return mapProtoBufTrainConfigToBuilder(new NeuralNetworkBuilder(), solverParameter);
	}

	/**
	 * map the general parameter
	 *
	 * @param neuralNetworkBuilder
	 * @param solverParameter
	 * @return
	 */
	public static NeuralNetworkBuilder mapProtoBufTrainConfigToBuilder(NeuralNetworkBuilder neuralNetworkBuilder, ProtoBufWrapper.SolverParameter solverParameter)
	{
		if (neuralNetworkBuilder == null)
		{
			throw new IllegalArgumentException("neuralNetworkBuilder must be not null!");
		}

		if (solverParameter == null)
		{
			throw new IllegalArgumentException("solverParameter must be not null!");
		}

		if (!solverParameter.hasBaseLr())
		{
			throw new IllegalArgumentException("The file must have a basic learning rate (base_lr)!");
		}
		neuralNetworkBuilder.setLearningRate(solverParameter.getBaseLr());

		if (solverParameter.hasMomentum())
		{
			neuralNetworkBuilder.setMomentum(solverParameter.getMomentum());
		}

		if (solverParameter.hasMaxIter())
		{
			neuralNetworkBuilder.setEpochs(solverParameter.getMaxIter());
		}

		return neuralNetworkBuilder;
	}

}
