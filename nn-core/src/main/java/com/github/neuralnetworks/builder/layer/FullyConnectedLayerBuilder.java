package com.github.neuralnetworks.builder.layer;

import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.LayerUtil;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.layer.structure.BiasLayerConnectable;
import com.github.neuralnetworks.builder.layer.structure.DropOutableLayer;
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
public class FullyConnectedLayerBuilder extends NamedSingleInputLayerBuilder implements BiasLayerConnectable, LearnableLayer, DropOutableLayer, MainFunctionsChangeable
{

	private ActivationType activationType = ActivationType.Nothing;
	private TransferFunctionType transferFunctionType = TransferFunctionType.WeightedSum;

	private int neuronNumber;
	private boolean addBias = true;

	// hyperparameters
	private float learningRate = -1;
	private float momentum = -1;
	private float l1weightDecay = -1;
	private float l2weightDecay = -1;
	private float dropoutRate = -1;

	private float biasLearningRate = -1;
	private float biasMomentum = -1;
	private float biasL1weightDecay = -1;
	private float biasL2weightDecay = -1;

	private RandomInitializer weightInitializer = null;
	private RandomInitializer biasWeightInitializer = null;

	public FullyConnectedLayerBuilder(int neuronNumber)
	{
		super("FullyConnectedLayer");

		if (neuronNumber < 1)
		{
			throw new IllegalStateException("neuron number must be greater than 0! (know " + neuronNumber + ")");
		}

		this.neuronNumber = neuronNumber;
	}


	@Override
	public Layer build(NeuralNetworkImpl neuralNetwork, String newLayerName, Layer inputLayer, Hyperparameters hyperparameters)
	{
		// initialize and add layer
		ConnectionFactory cf = neuralNetwork.getProperties().getParameter(Constants.CONNECTION_FACTORY);

		Layer newLayer = null;
		Layer biasLayer = null;

		FullyConnected fullyConnected = cf.fullyConnected(inputLayer, newLayer = new Layer(), inputLayer.getNeuronCount(), neuronNumber);

		newLayer.setLayerDimension(new int[] { neuronNumber });
		newLayer.setName(newLayerName);

		if (weightInitializer != null)
		{
			WeightInitializerFactory.initializeWeights(fullyConnected, weightInitializer);
		}

		if (neuralNetwork.getLayerCalculator() != null)
		{
			// change transfer and activation function
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

		FullyConnected biasConnection = null;
		if (addBias && neuralNetwork.getLayerCalculator() instanceof LayerCalculatorImpl)
		{
			biasConnection = cf.fullyConnected(biasLayer = new Layer(), newLayer, 1, neuronNumber);
			((LayerCalculatorImpl) neuralNetwork.getLayerCalculator()).addConnectionCalculator(biasLayer, new ConstantConnectionCalculator());
			biasLayer.setLayerDimension(new int[] { neuronNumber });
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
				hyperparameters.setLearningRate(fullyConnected, learningRate);
			}
			if (momentum != -1)
			{
				hyperparameters.setMomentum(fullyConnected, momentum);
			}
			if (l1weightDecay != -1)
			{
				hyperparameters.setL1WeightDecay(fullyConnected, l1weightDecay);
			}
			if (l2weightDecay != -1)
			{
				hyperparameters.setL2WeightDecay(fullyConnected, l2weightDecay);
			}
			if (dropoutRate != -1)
			{
				hyperparameters.setDropoutRate(fullyConnected, dropoutRate);
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

	@Override
	public boolean isAddBias()
	{
		return addBias;
	}

	@Override
	public void setAddBias(boolean addBias)
	{
		this.addBias = addBias;
	}

	public TransferFunctionType getTransferFunctionType()
	{
		return transferFunctionType;
	}

	@Override
	public FullyConnectedLayerBuilder setTransferFunctionType(TransferFunctionType transferFunctionType)
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

	@Override
	public FullyConnectedLayerBuilder setActivationType(ActivationType activationType)
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

	@Override
	public String toString()
	{
		return "FullyConnectedLayerBuilder{" +
				"activationType=" + activationType +
				", transferFunctionType=" + transferFunctionType +
				", neuronNumber=" + neuronNumber +
				", addBias=" + addBias +
				", learningRate=" + learningRate +
				", momentum=" + momentum +
				", l1weightDecay=" + l1weightDecay +
				", l2weightDecay=" + l2weightDecay +
				", dropoutRate=" + dropoutRate +
				", biasLearningRate=" + biasLearningRate +
				", biasMomentum=" + biasMomentum +
				", biasL1weightDecay=" + biasL1weightDecay +
				", biasL2weightDecay=" + biasL2weightDecay +
				", weightInitializer=" + weightInitializer +
				", biasWeightInitializer=" + biasWeightInitializer +
				'}';
	}
}
