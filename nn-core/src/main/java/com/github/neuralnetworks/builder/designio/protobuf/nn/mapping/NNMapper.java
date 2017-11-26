package com.github.neuralnetworks.builder.designio.protobuf.nn.mapping;

import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.designio.protobuf.nn.NNProtoBufWrapper;
import com.github.neuralnetworks.builder.layer.ConvolutionalLayerBuilder;
import com.github.neuralnetworks.builder.layer.DropoutLayerBuilder;
import com.github.neuralnetworks.builder.layer.FullyConnectedLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.LRNLayerBuilder;
import com.github.neuralnetworks.builder.layer.PoolingLayerBuilder;
import com.github.neuralnetworks.builder.layer.ReluLayerBuilder;
import com.github.neuralnetworks.builder.layer.SigmoidLayerBuilder;
import com.github.neuralnetworks.builder.layer.SoftmaxLayerBuilder;
import com.github.neuralnetworks.builder.layer.structure.BiasLayerConnectable;
import com.github.neuralnetworks.builder.layer.structure.KernelUsageOptions;
import com.github.neuralnetworks.builder.layer.structure.LayerBuilder;
import com.github.neuralnetworks.builder.layer.structure.LearnableLayer;
import com.github.neuralnetworks.builder.layer.structure.NamedSingleInputLayerBuilder;
import com.github.neuralnetworks.training.random.DummyFixedInitializer;
import com.github.neuralnetworks.training.random.NormalDistributionInitializer;
import com.github.neuralnetworks.training.random.RandomInitializer;
import com.github.neuralnetworks.util.Pair;

/**
 * @author tmey
 */
public class NNMapper {

	private static final Logger logger = LoggerFactory.getLogger(NNMapper.class);

	public static NeuralNetworkBuilder mapProtoBufNetConfigToBuilder(NNProtoBufWrapper.NetConfiguration netParameter) {
		return mapProtoBufNetConfigToBuilder(new NeuralNetworkBuilder(), netParameter);
	}

	/**
	 * map neural network configuration to the builder
	 *
	 * @param neuralNetworkBuilder
	 * @param netParameter
	 * @return
	 */
	public static NeuralNetworkBuilder mapProtoBufNetConfigToBuilder(NeuralNetworkBuilder neuralNetworkBuilder,
			NNProtoBufWrapper.NetConfiguration netParameter) {
		if (neuralNetworkBuilder == null) {
			throw new IllegalArgumentException("neuralNetworkBuilder must be not null!");
		}

		if (netParameter == null) {
			throw new IllegalArgumentException("netParameter must be not null!");
		}

		// create input layer
		if (netParameter.getInputDimCount() != 3) {
			throw new IllegalArgumentException(
					"the number of input dim must be 3! (it is " + (netParameter.getInputDimCount()) + ")");
		}

		String inputLayerName = "InputLayer";
		if (netParameter.hasInputName()) {
			netParameter.getInputName();
		}

		InputLayerBuilder inputLayerBuilder = new InputLayerBuilder(inputLayerName, netParameter.getInputDim(1),
				netParameter.getInputDim(2), netParameter.getInputDim(0));
		neuralNetworkBuilder.addLayerBuilder(inputLayerBuilder);

		// get default learn parameter
		if (netParameter.hasDefaultLearnParam()) {
			NNProtoBufWrapper.LearnParameter defaultLearnParam = netParameter.getDefaultLearnParam();

			if (defaultLearnParam.hasLr()) {
				neuralNetworkBuilder.setLearningRate(defaultLearnParam.getLr());
			}
			if (defaultLearnParam.hasWeightDecay()) {
				neuralNetworkBuilder.setL1weightDecay(defaultLearnParam.getWeightDecay());
			}
			if (defaultLearnParam.hasMomentum()) {
				neuralNetworkBuilder.setMomentum(defaultLearnParam.getMomentum());
			}
		}

		// create infrastructure
		Map<String, NNProtoBufWrapper.LayerParameter> mapNameToLayerParameter = new HashMap<>();

		// first check all layers and create structures
		for (NNProtoBufWrapper.LayerParameter layerParameter : netParameter.getLayersList()) {

			if (!layerParameter.hasType()) {
				throw new IllegalArgumentException("each layer need a type! - " + layerParameter.getName());
			}

			if (layerParameter.hasName()) {
				if (mapNameToLayerParameter.put(layerParameter.getName(), layerParameter) != null) {
					throw new IllegalArgumentException(
							"Each layer name must be unique! Please check " + layerParameter.getName() + "!");
				}

				if (layerParameter.getInputCount() > 0) {

					if (layerParameter.getInputCount() > 1) {
						throw new IllegalArgumentException("More than one input isn't supported at the moment!");
					}
				}
			}
		}

		// parse layer configuration
		for (int i = 0; i < netParameter.getLayersList().size(); i++) {
			NNProtoBufWrapper.LayerParameter currentLayer = netParameter.getLayersList().get(i);

			LayerBuilder layerBuilder = null;

			switch (currentLayer.getType()) {
			case CONVOLUTIONAL: {
				if (!currentLayer.hasConvParam()) {
					throw new IllegalArgumentException(
							"The parameter for the convolutional layer (" + i + 1 + ") are missing!");
				}
				NNProtoBufWrapper.ConvolutionParameter convolutionParam = currentLayer.getConvParam();

				if (!convolutionParam.hasNumKernel()) {
					throw new IllegalArgumentException("number of kernels is missing " + i + 1 + " layer! (NumOutput)");
				}

				if (!convolutionParam.hasKernelParam()) {
					throw new IllegalArgumentException("kernel parameters are missing in " + i + 1 + " layer!");
				}

				NNProtoBufWrapper.KernelParameter kernelParam = convolutionParam.getKernelParam();

				Pair<Integer, Integer> kernelSize = getKernelSize(kernelParam);

				if (kernelSize == null) {
					throw new IllegalArgumentException("the kernel size is missing in " + i + 1 + " layer!");
				}

				ConvolutionalLayerBuilder convolutionalLayerBuilder = new ConvolutionalLayerBuilder(
						kernelSize.getLeft(), kernelSize.getRight(), convolutionParam.getNumKernel());

				addInputAndLayerName(convolutionalLayerBuilder, currentLayer);

				addStrideAndPadding(convolutionalLayerBuilder, kernelParam);

				boolean useBias = true;

				if (convolutionParam.hasBiasTerm()) {
					useBias = convolutionParam.getBiasTerm();
					convolutionalLayerBuilder.setAddBias(useBias);
				}

				// initialize learning rates

				NNProtoBufWrapper.LearnParameter mainLearnParameter = null;
				NNProtoBufWrapper.LearnParameter biasLearnParameter = null;

				if (convolutionParam.hasLearnParam()) {
					mainLearnParameter = convolutionParam.getLearnParam();
				}
				if (convolutionParam.hasBiasLearnParam()) {
					mainLearnParameter = convolutionParam.getBiasLearnParam();
				}

				parseLearningParameter(convolutionalLayerBuilder, mainLearnParameter, biasLearnParameter,
						neuralNetworkBuilder.getLearningRate(), neuralNetworkBuilder.getL1weightDecay(),
						neuralNetworkBuilder.getMomentum(), useBias);

				// weight initializer

				if (convolutionParam.hasWeightFiller()) {
					convolutionalLayerBuilder
							.setWeightInitializer(getWeightIninitializer(convolutionParam.getWeightFiller()));
				}

				if (convolutionParam.hasBiasFiller()) {
					convolutionalLayerBuilder
							.setBiasWeightInitializer(getWeightIninitializer(convolutionParam.getBiasFiller()));
				}

				layerBuilder = convolutionalLayerBuilder;
				break;
			}
			case POOLING: {
				if (!currentLayer.hasPoolParam()) {
					throw new IllegalArgumentException(
							"The parameter for the pooling layer (" + i + 1 + ". layer) are missing!");
				}
				NNProtoBufWrapper.PoolParameter poolingParam = currentLayer.getPoolParam();

				if (!poolingParam.hasKernelParam()) {
					throw new IllegalArgumentException("kernel parameters are missing in " + i + 1 + " layer!");
				}

				NNProtoBufWrapper.KernelParameter kernelParam = poolingParam.getKernelParam();

				Pair<Integer, Integer> kernelSize = getKernelSize(kernelParam);

				if (kernelSize == null) {
					throw new IllegalArgumentException("the kernel size is missing in " + i + 1 + " layer!");
				}

				PoolingLayerBuilder poolingLayerBuilderLayerBuilder = new PoolingLayerBuilder(kernelSize.getLeft(),
						kernelSize.getRight());

				addInputAndLayerName(poolingLayerBuilderLayerBuilder, currentLayer);

				addStrideAndPadding(poolingLayerBuilderLayerBuilder, kernelParam);

				if (poolingParam.hasType()) {

					switch (poolingParam.getType()) {
					case MAX:
						poolingLayerBuilderLayerBuilder.setTransferFunctionType(TransferFunctionType.Max_Polling2D);
						break;
					case AVE:
						poolingLayerBuilderLayerBuilder.setTransferFunctionType(TransferFunctionType.Average_Pooling2D);
						break;
					default:
						throw new IllegalArgumentException(
								"Unknown pooling type in " + i + 1 + ". layer! " + poolingParam.getType());
					}

				}
				layerBuilder = poolingLayerBuilderLayerBuilder;
				break;

			}
			case FULLY: {
				if (!currentLayer.hasFullyParam()) {
					throw new IllegalArgumentException(
							"The parameter for the fully connected layer (" + i + 1 + ". layer) are missing!");
				}
				NNProtoBufWrapper.FullyConnectedParameter innerProductParam = currentLayer.getFullyParam();

				if (!innerProductParam.hasNumNeurons()) {
					throw new IllegalArgumentException(
							"The number of neurons (numNeurons) for the fully connected layer (" + i + 1
									+ ". layer) are missing!");
				}

				// build layer
				FullyConnectedLayerBuilder fullyConnectedLayerBuilder = new FullyConnectedLayerBuilder(
						innerProductParam.getNumNeurons());

				addInputAndLayerName(fullyConnectedLayerBuilder, currentLayer);

				boolean useBias = true;

				if (innerProductParam.hasBiasTerm()) {
					useBias = innerProductParam.getBiasTerm();
					fullyConnectedLayerBuilder.setAddBias(useBias);
				}

				// initialize learning rates

				NNProtoBufWrapper.LearnParameter mainLearnParameter = null;
				NNProtoBufWrapper.LearnParameter biasLearnParameter = null;

				if (innerProductParam.hasLearnParam()) {
					mainLearnParameter = innerProductParam.getLearnParam();
				}
				if (innerProductParam.hasBiasLearnParam()) {
					mainLearnParameter = innerProductParam.getBiasLearnParam();
				}

				parseLearningParameter(fullyConnectedLayerBuilder, mainLearnParameter, biasLearnParameter,
						neuralNetworkBuilder.getLearningRate(), neuralNetworkBuilder.getL1weightDecay(),
						neuralNetworkBuilder.getMomentum(), useBias);

				// weight initializer

				if (innerProductParam.hasWeightFiller()) {
					fullyConnectedLayerBuilder
							.setWeightInitializer(getWeightIninitializer(innerProductParam.getWeightFiller()));
				}

				if (innerProductParam.hasBiasFiller()) {
					fullyConnectedLayerBuilder
							.setBiasWeightInitializer(getWeightIninitializer(innerProductParam.getBiasFiller()));
				}

				layerBuilder = fullyConnectedLayerBuilder;
				break;
			}

			// activation functions

			case RELU: {
				ReluLayerBuilder functionLayerBuilder = new ReluLayerBuilder();

				try {
					addInputAndLayerName(functionLayerBuilder, currentLayer);
				} catch (IllegalArgumentException e) {
					throw new IllegalArgumentException("Problem in the " + (i + 1) + ". layer (Relu)!", e);
				}

				layerBuilder = functionLayerBuilder;
				break;
			}
			case SIGMOID: {
				SigmoidLayerBuilder functionLayerBuilder = new SigmoidLayerBuilder();
				try {
					addInputAndLayerName(functionLayerBuilder, currentLayer);
				} catch (IllegalArgumentException e) {
					throw new IllegalArgumentException("Problem in the " + (i + 1) + ". layer (Sigmoid)!", e);
				}
				layerBuilder = functionLayerBuilder;
				break;
			}
			case SOFTMAX: {
				SoftmaxLayerBuilder functionLayerBuilder = new SoftmaxLayerBuilder();
				try {
					addInputAndLayerName(functionLayerBuilder, currentLayer);
				} catch (IllegalArgumentException e) {
					throw new IllegalArgumentException("Problem in the " + (i + 1) + ". layer (Softmax)!", e);
				}
				layerBuilder = functionLayerBuilder;
				break;
			}

			// special layer
			case DROPOUT: {
				DropoutLayerBuilder functionLayerBuilder = new DropoutLayerBuilder();
				try {
					addInputAndLayerName(functionLayerBuilder, currentLayer);
				} catch (IllegalArgumentException e) {
					throw new IllegalArgumentException("Problem in the " + (i + 1) + ". layer (Dropout)!", e);
				}

				if (currentLayer.hasDropoutRate()) {
					functionLayerBuilder.setRate(currentLayer.getDropoutRate());
				}

				layerBuilder = functionLayerBuilder;
				break;
			}

			// Mixed
			case LRN: {
				LRNLayerBuilder lrnLayerBuilder = new LRNLayerBuilder();

				layerBuilder = lrnLayerBuilder;
				break;
			}

			default: {
				throw new IllegalArgumentException(
						"Unsupported layer type " + currentLayer.getType() + " in " + (i + 1) + ". layer!");
			}
			}

			// add layer builder to network builder
			neuralNetworkBuilder.addLayerBuilder(layerBuilder);

			// // get next layer
			// if (!mapBottomToLayerParameter.containsKey(layerName))
			// {
			// currentLayer = null;
			// } else
			// {
			// currentLayer = mapBottomToLayerParameter.get(layerName);
			// mapBottomToLayerParameter.remove(layerName);
			// }
		}

		return neuralNetworkBuilder;

	}

	private static void addInputAndLayerName(NamedSingleInputLayerBuilder layerBuilder,
			NNProtoBufWrapper.LayerParameter layerParameter) throws IllegalArgumentException {
		if (layerParameter.hasName()) {
			layerBuilder.setName(layerParameter.getName());
		}

		if (layerParameter.getInputCount() > 0) {
			if (layerParameter.getInputCount() > 1) {
				throw new IllegalArgumentException(
						"More than one input isn't supported for this layer! (current inputs: "
								+ layerParameter.getInputCount() + ")");
			}
			layerBuilder.setInputLayerName(layerParameter.getInput(0));
		}
	}

	private static void addStrideAndPadding(LayerBuilder layerBuilder,
			NNProtoBufWrapper.KernelParameter kernelParameter) {
		if (layerBuilder instanceof KernelUsageOptions) {
			KernelUsageOptions kernelUsageOptions = (KernelUsageOptions) layerBuilder;
			// extract stride
			Pair<Integer, Integer> kernelStride = getKernelStride(kernelParameter);

			if (kernelStride != null) {
				kernelUsageOptions.setStrideRows(kernelStride.getLeft());
				kernelUsageOptions.setStrideColumns(kernelStride.getRight());
			}

			// extract padding
			Pair<Integer, Integer> kernelPadding = getKernelPadding(kernelParameter);

			if (kernelPadding != null) {
				kernelUsageOptions.setPaddingRows(kernelPadding.getLeft());
				kernelUsageOptions.setPaddingColumns(kernelPadding.getRight());
			}
		}

	}

	private static Pair<Integer, Integer> getKernelSize(NNProtoBufWrapper.KernelParameter kernelParameter) {
		if (kernelParameter.hasKernelSize()) {
			if (kernelParameter.hasKernelH()) {
				logger.warn(
						"kernelH: " + kernelParameter.getKernelH() + " will be ignored, because kernel size was set!");
			}
			if (kernelParameter.hasKernelW()) {
				logger.warn(
						"kernelW: " + kernelParameter.getKernelW() + " will be ignored, because kernel size was set!");
			}
			return new Pair<>(kernelParameter.getKernelSize(), kernelParameter.getKernelSize());
		}

		if (kernelParameter.hasKernelH() && kernelParameter.hasKernelW()) {
			return new Pair<>(kernelParameter.getKernelH(), kernelParameter.getKernelW());
		}

		if (kernelParameter.hasKernelH()) {
			logger.warn("kernelH: " + kernelParameter.getKernelH() + " will be ignored, because kernelW isn't used!");
		}
		if (kernelParameter.hasKernelW()) {
			logger.warn("kernelW: " + kernelParameter.getKernelW() + " will be ignored, because kernelH isn't used!");
		}
		return null;
	}

	private static Pair<Integer, Integer> getKernelPadding(NNProtoBufWrapper.KernelParameter kernelParameter) {
		if (kernelParameter.hasPadding()) {
			if (kernelParameter.hasPaddingH()) {
				logger.warn("PaddingH: " + kernelParameter.getPaddingH()
						+ " will be ignored, because padding size was set!");
			}
			if (kernelParameter.hasPaddingW()) {
				logger.warn("PaddingW: " + kernelParameter.getPaddingW()
						+ " will be ignored, because padding size was set!");
			}
			return new Pair<>(kernelParameter.getPadding(), kernelParameter.getPadding());
		}

		if (kernelParameter.hasPaddingH() && kernelParameter.hasPaddingW()) {
			return new Pair<>(kernelParameter.getPaddingH(), kernelParameter.getPaddingW());
		}

		if (kernelParameter.hasPaddingH()) {
			logger.warn(
					"PaddingH: " + kernelParameter.getPaddingH() + " will be ignored, because paddingW isn't used!");
		}
		if (kernelParameter.hasPaddingW()) {
			logger.warn(
					"PaddingW: " + kernelParameter.getPaddingW() + " will be ignored, because paddingH isn't used!");
		}
		return null;
	}

	private static Pair<Integer, Integer> getKernelStride(NNProtoBufWrapper.KernelParameter kernelParameter) {
		if (kernelParameter.hasStride()) {
			if (kernelParameter.hasStrideH()) {
				logger.warn(
						"StrideH: " + kernelParameter.getStrideH() + " will be ignored, because stride size was set!");
			}
			if (kernelParameter.hasStrideW()) {
				logger.warn(
						"StrideW: " + kernelParameter.getStrideW() + " will be ignored, because stride size was set!");
			}
			return new Pair<>(kernelParameter.getStride(), kernelParameter.getStride());
		}

		if (kernelParameter.hasStrideH() && kernelParameter.hasStrideW()) {
			return new Pair<>(kernelParameter.getStrideH(), kernelParameter.getStrideW());
		}

		if (kernelParameter.hasStrideH()) {
			logger.warn("StrideH: " + kernelParameter.getStrideH() + " will be ignored, because strideW isn't used!");
		}
		if (kernelParameter.hasStrideW()) {
			logger.warn("StrideW: " + kernelParameter.getStrideW() + " will be ignored, because strideH isn't used!");
		}
		return null;
	}

	/**
	 * set the learning rate and weight decay for current layer and bias layer
	 *
	 * @param layerBuilder
	 * @param learnParameter
	 */
	public static void parseLearningParameter(LayerBuilder layerBuilder,
			NNProtoBufWrapper.LearnParameter learnParameter, NNProtoBufWrapper.LearnParameter biasLearnParameter,
			float generalLearningRage, float generalL1WeightDecay, float generalMomentum, boolean useBias) {

		if (layerBuilder instanceof LearnableLayer && learnParameter != null) {
			LearnableLayer learnableLayer = (LearnableLayer) layerBuilder;
			float localGeneralL1WeightDecay = generalL1WeightDecay;
			float localGeneralLearningRate = generalLearningRage;
			float localGeneralMomentum1 = generalMomentum;

			if (learnParameter.getAbsoluteValues()) {
				localGeneralL1WeightDecay = 1;
				localGeneralLearningRate = 1;
				localGeneralMomentum1 = 1;
			}

			if (learnParameter.hasLr()) {
				learnableLayer.setLearningRate(localGeneralLearningRate * learnParameter.getLr());
			}
			if (learnParameter.hasWeightDecay()) {
				learnableLayer.setL1weightDecay(localGeneralL1WeightDecay * learnParameter.getWeightDecay());
			}
			if (learnParameter.hasMomentum()) {
				learnableLayer.setMomentum(localGeneralMomentum1 * learnParameter.getMomentum());
			}
		}

		if (layerBuilder instanceof BiasLayerConnectable && useBias && biasLearnParameter != null) {
			float localGeneralLearningRate = generalLearningRage;
			float localGeneralL1WeightDecay = generalL1WeightDecay;
			float localGeneralMomentum1 = generalMomentum;

			if (biasLearnParameter.getAbsoluteValues()) {
				localGeneralL1WeightDecay = 1;
				localGeneralLearningRate = 1;
				localGeneralMomentum1 = 1;
			}

			if (biasLearnParameter.hasLr()) {
				((BiasLayerConnectable) layerBuilder)
						.setBiasLearningRate(localGeneralLearningRate * biasLearnParameter.getLr());
			}
			if (biasLearnParameter.hasWeightDecay()) {
				((BiasLayerConnectable) layerBuilder)
						.setBiasL1weightDecay(localGeneralL1WeightDecay * biasLearnParameter.getWeightDecay());
			}
			if (biasLearnParameter.hasMomentum()) {
				((BiasLayerConnectable) layerBuilder)
						.setBiasMomentum(localGeneralMomentum1 * biasLearnParameter.getMomentum());
			}
		}
	}

	private static RandomInitializer getWeightIninitializer(NNProtoBufWrapper.FillerParameter weightFiller) {
		if (!weightFiller.hasType()) {
			throw new IllegalArgumentException("Each weight filler needs a type!");
		}

		switch (weightFiller.getType()) {
		case CONSTANT: {
			float value = 0;
			if (weightFiller.hasValue()) {
				value = weightFiller.getValue();
			}

			return new DummyFixedInitializer(value);
		}
		case GAUSSIAN:
			float std;
			float mean = 0;

			if (weightFiller.hasGaussParam()) {
				NNProtoBufWrapper.GaussDistrParameter gaussParam = weightFiller.getGaussParam();
				if (!gaussParam.hasStd()) {
					throw new IllegalArgumentException("A gaussian initializer needs the standard deviation (std)!");
				}
				std = gaussParam.getStd();

				if (gaussParam.hasMean()) {
					mean = gaussParam.getMean();
				}
			} else {
				throw new IllegalArgumentException("The gaussian initializer need GaussParam!");
			}

			return new NormalDistributionInitializer(mean, std);
		default:
			throw new IllegalArgumentException("Unknown weight filler type: " + weightFiller.getType());
		}
	}

}
