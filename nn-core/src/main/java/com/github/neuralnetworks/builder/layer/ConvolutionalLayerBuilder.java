package com.github.neuralnetworks.builder.layer;

import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.LayerUtil;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.layer.structure.BiasLayerConnectable;
import com.github.neuralnetworks.builder.layer.structure.DropOutableLayer;
import com.github.neuralnetworks.builder.layer.structure.KernelUsageOptions;
import com.github.neuralnetworks.builder.layer.structure.LearnableLayer;
import com.github.neuralnetworks.builder.layer.structure.MainFunctionsChangeable;
import com.github.neuralnetworks.builder.layer.structure.NamedSingleInputLayerBuilder;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.cpu.ConstantConnectionCalculator;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.training.random.RandomInitializer;
import com.github.neuralnetworks.training.random.WeightInitializerFactory;
import com.github.neuralnetworks.util.Constants;

/**
 * @author tmey
 */
public class ConvolutionalLayerBuilder extends NamedSingleInputLayerBuilder implements LearnableLayer, BiasLayerConnectable, DropOutableLayer, MainFunctionsChangeable, KernelUsageOptions
{

	/**
	 * 0 means stride = filterSize
	 */
	private int strideRows = 0;
	private int strideColumns = 0;
	private int paddingRows = 0;
	private int paddingColumns = 0;
	private int filterRows = 3;
	private int filterColumns = 3;
	private int featureMaps = 1;

	private ActivationType activationType = ActivationType.Nothing;
	private TransferFunctionType transferFunctionType = TransferFunctionType.Conv2D;

	// hyper parameters
	private float learningRate = -1;
	private float momentum = -1;
	private float l1weightDecay = -1;
	private float l2weightDecay = -1;
	private float dropoutRate = -1;

	private boolean addBias = true;
	private float biasLearningRate = -1;
	private float biasMomentum = -1;
	private float biasL1weightDecay = -1;
	private float biasL2weightDecay = -1;

	private RandomInitializer weightInitializer = null;
	private RandomInitializer biasWeightInitializer = null;

	public ConvolutionalLayerBuilder(int filterSize, int featureMaps)
	{
		super("ConvolutionalLayer");

		this.setFilterSize(filterSize);
		this.setFeatureMaps(featureMaps);
	}

	public ConvolutionalLayerBuilder(int filterSizeH, int filterSizeW, int featureMaps)
	{
		super("ConvolutionalLayer");

		this.setFilterColumns(filterSizeW);
		this.setFilterRows(filterSizeH);
		this.setFeatureMaps(featureMaps);
	}

	@Override
	protected Layer build(NeuralNetworkImpl neuralNetwork, String newLayerName, Layer inputLayer, Hyperparameters hyperparameters)
	{

		ConnectionFactory cf = neuralNetwork.getProperties().getParameter(Constants.CONNECTION_FACTORY);

		Layer newLayer = new Layer();
		Layer biasLayer = null;

		// initialize default parameter the right way
		int localStrideRows = strideRows;
		int localStrideColumns = strideColumns;

		if (localStrideColumns == 0)
		{
			localStrideColumns = 1;
		}

		if (localStrideRows == 0)
		{
			localStrideRows = 1;
		}

		// search last connection
		int inputFMRows = -1;
		int inputFMCols = -1;
		int inputFilters = -1;

		int[] layerDimension = inputLayer.getLayerDimension();
		if (layerDimension.length != 3)
		{
			throw new IllegalStateException("The current layer should be connected to a not 3 dimensional layer (" + layerDimension.length + ")");
		}

		inputFMRows = layerDimension[0];
		inputFMCols = layerDimension[1];
		inputFilters = layerDimension[2];

		if (inputFMRows < 1 || inputFMCols < 1 || inputFilters < 1)
		{
			throw new IllegalStateException("The inputLayer is");
		}

		// connect layer
		Conv2DConnection convConnection = cf.conv2d(inputLayer, newLayer, inputFMRows, inputFMCols, inputFilters,
				filterRows, filterColumns, featureMaps, localStrideRows, localStrideColumns, paddingRows, paddingColumns);

		newLayer.setLayerDimension(new int[] { convConnection.getOutputFeatureMapRowsWithPadding(), convConnection.getOutputFeatureMapColumnsWithPadding(), convConnection.getOutputFilters() });
		newLayer.setName(newLayerName);

		if (weightInitializer != null)
		{
			WeightInitializerFactory.initializeWeights(convConnection, weightInitializer);
		}

		if (neuralNetwork.getLayerCalculator() != null)
		{
			LayerUtil.changeActivationAndTransferFunction(neuralNetwork.getLayerCalculator(), newLayer, transferFunctionType, activationType);

			// activate drop out
			if (this.dropoutRate > 0)
			{
				LayerCalculatorImpl lc = (LayerCalculatorImpl) neuralNetwork.getLayerCalculator();
				if (lc.getConnectionCalculator(newLayer) instanceof ConnectionCalculatorImpl)
				{
					ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) lc.getConnectionCalculator(newLayer);
					cc.addActivationFunction(OperationsFactory.noiseMask(this.dropoutRate, 0));
				}
			}
		}

		Conv2DConnection biasConnection = null;
		if (addBias)
		{
			biasConnection = cf.conv2d(biasLayer = new Layer(), newLayer, convConnection.getOutputFeatureMapRows(), convConnection.getOutputFeatureMapColumns(), 1, 1, 1, featureMaps, 1,
					1,
					paddingRows, paddingColumns);
			if (neuralNetwork.getLayerCalculator() != null)
			{
				((LayerCalculatorImpl) neuralNetwork.getLayerCalculator()).addConnectionCalculator(biasLayer, new ConstantConnectionCalculator());
			}
			biasLayer.setLayerDimension(new int[] { newLayer.getNeuronCount() });
			biasLayer.setName("bias_layer_to_" + newLayerName);

			if (biasWeightInitializer != null)
			{
				WeightInitializerFactory.initializeWeights(biasConnection, biasWeightInitializer);
			}
		}

		// add new layer to the network
		neuralNetwork.addLayer(newLayer);
		if (biasLayer != null)
		{
			neuralNetwork.addLayer(biasLayer);
		}

		// set hyperparameters
		if (hyperparameters != null)
		{
			if (learningRate != -1)
			{
				hyperparameters.setLearningRate(convConnection, learningRate);
			}
			if (momentum != -1)
			{
				hyperparameters.setMomentum(convConnection, momentum);
			}
			if (l1weightDecay != -1)
			{
				hyperparameters.setL1WeightDecay(convConnection, l1weightDecay);
			}
			if (l2weightDecay != -1)
			{
				hyperparameters.setL2WeightDecay(convConnection, l2weightDecay);
			}

			// bias layer
			if (biasConnection != null)
			{
				if (biasLearningRate != -1)
				{
					hyperparameters.setLearningRate(biasConnection, biasLearningRate);
				}
				if (biasMomentum != -1)
				{
					hyperparameters.setMomentum(biasConnection, biasMomentum);
				}
				if (biasL1weightDecay != -1)
				{
					hyperparameters.setL1WeightDecay(biasConnection, biasL1weightDecay);
				}
				if (biasL2weightDecay != -1)
				{
					hyperparameters.setL2WeightDecay(biasConnection, biasL2weightDecay);
				}
			}
		}

		return newLayer;
	}

	public void setFilterSize(int size)
	{
		if (size <= 0)
		{
			throw new IllegalArgumentException("The filter size must be greater than 0!");
		}

		this.filterColumns = size;
		this.filterRows = size;
	}

	public void setPaddingSize(int size)
	{
//		if (size <= 0)
//		{
//			throw new IllegalArgumentException("The padding size must be greater than 0!");
//		}

		this.paddingColumns = size;
		this.paddingRows = size;
	}

	public void setStrideSize(int size)
	{
		if (size <= 0)
		{
			throw new IllegalArgumentException("The stride size must be greater than 0!");
		}

		this.strideColumns = size;
		this.strideRows = size;
	}

	public void setFeatureMaps(int featureMaps)
	{
		if (featureMaps <= 0)
		{
			throw new IllegalArgumentException("The number of featureMaps must be greater than 0!");
		}

		this.featureMaps = featureMaps;
	}

	public TransferFunctionType getTransferFunctionType()
	{
		return transferFunctionType;
	}

	public ConvolutionalLayerBuilder setTransferFunctionType(TransferFunctionType transferFunctionType)
	{
		if (transferFunctionType == null)
		{
			throw new IllegalArgumentException("transferFunctionType must be not null!");
		}

		this.transferFunctionType = transferFunctionType;
		return this;
	}

	public ActivationType getActivationType()
	{
		return activationType;
	}

	public ConvolutionalLayerBuilder setActivationType(ActivationType activationType)
	{
		if (activationType == null)
		{
			throw new IllegalArgumentException("activationType must be not null!");
		}

		this.activationType = activationType;
		return this;
	}

	@Override
	public void setLearningRate(float learningRate)
	{
		if (learningRate <= 0)
		{
			throw new IllegalArgumentException("The learning rate must be greater than 0!");
		}

		this.learningRate = learningRate;
	}

	@Override
	public void setMomentum(float momentum)
	{
		if (momentum < 0)
		{
			throw new IllegalArgumentException("The momentum must be equals or greater than 0!");
		}

		this.momentum = momentum;
	}

	@Override
	public void setL1weightDecay(float l1weightDecay)
	{
		if (l1weightDecay < 0)
		{
			throw new IllegalArgumentException("The l1weightDecay must be equals or greater than 0!");
		}

		this.l1weightDecay = l1weightDecay;
	}

	@Override
	public void setL2weightDecay(float l2weightDecay)
	{
		if (l2weightDecay < 0)
		{
			throw new IllegalArgumentException("The l2weightDecay must be equals or greater than 0!");
		}

		this.l2weightDecay = l2weightDecay;
	}

	@Override
	public void setDropoutRate(float dropoutRate)
	{
		if (dropoutRate < 0)
		{
			throw new IllegalArgumentException("The dropoutRate must be equals or greater than 0!");
		}

		this.dropoutRate = dropoutRate;
	}

	@Override
	public boolean isAddBias()
	{
		return this.addBias;
	}

	@Override
	public void setAddBias(boolean addBias)
	{
		this.addBias = addBias;
	}

	@Override
	public RandomInitializer getWeightInitializer()
	{
		return weightInitializer;
	}

	@Override
	public void setWeightInitializer(RandomInitializer weightInitializer)
	{
		this.weightInitializer = weightInitializer;
	}

	@Override
	public RandomInitializer getBiasWeightInitializer()
	{
		return biasWeightInitializer;
	}

	@Override
	public void setBiasWeightInitializer(RandomInitializer biasWeightInitializer)
	{
		this.biasWeightInitializer = biasWeightInitializer;
	}

	public float getBiasLearningRate()
	{
		return biasLearningRate;
	}

	public void setBiasLearningRate(float biasLearningRate)
	{
		if (biasLearningRate <= 0)
		{
			throw new IllegalArgumentException("The biasLearningRate must be greater than 0!");
		}

		this.biasLearningRate = biasLearningRate;
	}

	public float getBiasMomentum()
	{
		return biasMomentum;
	}

	public void setBiasMomentum(float biasMomentum)
	{
		if (biasMomentum < 0)
		{
			throw new IllegalArgumentException("The biasMomentum must be equals or greater than 0!");
		}

		this.biasMomentum = biasMomentum;
	}

	public float getBiasL1weightDecay()
	{
		return biasL1weightDecay;
	}

	public void setBiasL1weightDecay(float biasL1weightDecay)
	{
		if (biasL1weightDecay < 0)
		{
			throw new IllegalArgumentException("The biasL1weightDecay must be equals or greater than 0!");
		}

		this.biasL1weightDecay = biasL1weightDecay;
	}

	public float getBiasL2weightDecay()
	{
		return biasL2weightDecay;
	}

	public void setBiasL2weightDecay(float biasL2weightDecay)
	{
		if (biasL2weightDecay < 0)
		{
			throw new IllegalArgumentException("The biasL2weightDecay must be equals or greater than 0!");
		}

		this.biasL2weightDecay = biasL2weightDecay;
	}

	public void setStrideRows(int strideRows)
	{
		if (strideRows <= 0)
		{
			throw new IllegalArgumentException("The strideRows must be greater than 0!");
		}
		this.strideRows = strideRows;
	}

	public void setStrideColumns(int strideColumns)
	{
		if (strideColumns <= 0)
		{
			throw new IllegalArgumentException("The strideColumns must be greater than 0!");
		}
		this.strideColumns = strideColumns;
	}

	public void setPaddingRows(int paddingRows)
	{
		if (paddingRows < 0)
		{
			throw new IllegalArgumentException("The paddingRows must be greater othan r equals 0!");
		}
		this.paddingRows = paddingRows;
	}

	public void setPaddingColumns(int paddingColumns)
	{
		if (paddingColumns < 0)
		{
			throw new IllegalArgumentException("The paddingColumns must be greater than 0!");
		}
		this.paddingColumns = paddingColumns;
	}

	public void setFilterRows(int filterRows)
	{
		if (filterRows <= 0)
		{
			throw new IllegalArgumentException("The filterRows must be greater than 0!");
		}
		this.filterRows = filterRows;
	}

	public void setFilterColumns(int filterColumns)
	{
		if (filterColumns <= 0)
		{
			throw new IllegalArgumentException("The filterColumns must be greater than 0!");
		}
		this.filterColumns = filterColumns;
	}

	@Override
	public String toString()
	{
		return "ConvolutionalLayerBuilder{" +
				"name=" + (getName() != null ? getName() : DEFAULT_LAYER_NAME) +
				", inputLayer=" + getInputLayerName() +
				", strideRows=" + strideRows +
				", strideColumns=" + strideColumns +
				", paddingRows=" + paddingRows +
				", paddingColumns=" + paddingColumns +
				", filterRows=" + filterRows +
				", filterColumns=" + filterColumns +
				", featureMaps=" + featureMaps +
				", activationType=" + activationType +
				", transferFunctionType=" + transferFunctionType +
				", learningRate=" + learningRate +
				", momentum=" + momentum +
				", l1weightDecay=" + l1weightDecay +
				", l2weightDecay=" + l2weightDecay +
				", dropoutRate=" + dropoutRate +
				", addBias=" + addBias +
				", biasLearningRate=" + biasLearningRate +
				", biasMomentum=" + biasMomentum +
				", biasL1weightDecay=" + biasL1weightDecay +
				", biasL2weightDecay=" + biasL2weightDecay +
				", weightInitializer=" + weightInitializer +
				", biasWeightInitializer=" + biasWeightInitializer +
				'}';
	}
}
