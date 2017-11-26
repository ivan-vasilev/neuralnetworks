package com.github.neuralnetworks.calculation;

import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.architecture.types.DBN;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.architecture.types.StackedAutoencoder;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.calculation.operations.cpu.ConstantConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.cpu.MaxoutWinners;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.util.Util;

/**
 * Factory for calculation providers (Aparapi, OpenCL)
 */
public class CalculationFactory
{

	public static LayerCalculatorImpl lcWeightedSum(NeuralNetworkImpl nn, TensorFunction outputActivation)
	{
		LayerCalculatorImpl lc = new LayerCalculatorImpl();
		for (Layer l : nn.getLayers())
		{
			if (!Util.isBias(l))
			{
				if (Util.isConvolutional(l))
				{
					ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) OperationsFactory.conv2D();
					if (l == nn.getOutputLayer() && outputActivation != null)
					{
						cc.addActivationFunction(outputActivation);
					}
					lc.addConnectionCalculator(l, cc);
				} else
				{
					ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) OperationsFactory.weightedSum();
					if (l == nn.getOutputLayer() && outputActivation != null)
					{
						cc.addActivationFunction(outputActivation);
					}
					lc.addConnectionCalculator(l, cc);
				}
			} else
			{
				lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
			}
		}

		return lc;
	}

	public static LayerCalculatorImpl lcSigmoid(NeuralNetworkImpl nn, TensorFunction outputActivation)
	{
		LayerCalculatorImpl lc = new LayerCalculatorImpl();
		for (Layer l : nn.getLayers())
		{
			if (!Util.isBias(l))
			{
				if (Util.isConvolutional(l))
				{
					if (outputActivation != null && l == nn.getOutputLayer())
					{
						ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) OperationsFactory.conv2D();
						if (l == nn.getOutputLayer() && outputActivation != null)
						{
							cc.addActivationFunction(outputActivation);
						}
						lc.addConnectionCalculator(l, cc);
					} else
					{
						lc.addConnectionCalculator(l, OperationsFactory.conv2DSigmoid());
					}
				} else if (!Util.isSubsampling(l))
				{
					if (outputActivation != null && l == nn.getOutputLayer())
					{
						ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) OperationsFactory.weightedSum();
						if (l == nn.getOutputLayer() && outputActivation != null)
						{
							cc.addActivationFunction(outputActivation);
						}
						lc.addConnectionCalculator(l, cc);
					} else
					{
						lc.addConnectionCalculator(l, OperationsFactory.sigmoidConnectionCalculator());
					}
				}
			} else
			{
				lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
			}
		}

		return lc;
	}

	public static LayerCalculatorImpl lcMaxout(NeuralNetworkImpl nn)
	{
		LayerCalculatorImpl lc = new LayerCalculatorImpl();
		for (Layer l : nn.getLayers())
		{
			if (!Util.isBias(l) && !Util.isSubsampling(l) && !Util.isConvolutional(l))
			{
				lc.addConnectionCalculator(l, OperationsFactory.maxout());
			} else
			{
				lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
			}
		}

		return lc;
	}

	public static LayerCalculatorImpl lcSoftRelu(NeuralNetworkImpl nn, TensorFunction outputActivation)
	{
		LayerCalculatorImpl lc = new LayerCalculatorImpl();
		for (Layer l : nn.getLayers())
		{
			if (!Util.isBias(l))
			{
				if (Util.isConvolutional(l))
				{
					if (outputActivation != null && l == nn.getOutputLayer())
					{
						ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) OperationsFactory.conv2D();
						if (l == nn.getOutputLayer() && outputActivation != null)
						{
							cc.addActivationFunction(outputActivation);
						}
						lc.addConnectionCalculator(l, cc);
					} else
					{
						lc.addConnectionCalculator(l, OperationsFactory.conv2DSoftReLU());
					}
				} else
				{
					if (outputActivation != null && l == nn.getOutputLayer())
					{
						ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) OperationsFactory.weightedSum();
						if (l == nn.getOutputLayer() && outputActivation != null)
						{
							cc.addActivationFunction(outputActivation);
						}
						lc.addConnectionCalculator(l, cc);
					} else
					{
						lc.addConnectionCalculator(l, OperationsFactory.softReLUConnectionCalculator());
					}
				}
			} else
			{
				lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
			}
		}

		return lc;
	}

	public static LayerCalculatorImpl lcRelu(NeuralNetworkImpl nn, TensorFunction outputActivation)
	{
		LayerCalculatorImpl lc = new LayerCalculatorImpl();
		for (Layer l : nn.getLayers())
		{
			if (!Util.isBias(l))
			{
				if (Util.isConvolutional(l))
				{
					if (outputActivation != null && l == nn.getOutputLayer())
					{
						ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) OperationsFactory.conv2D();
						if (l == nn.getOutputLayer() && outputActivation != null)
						{
							cc.addActivationFunction(outputActivation);
						}
						lc.addConnectionCalculator(l, cc);
					} else
					{
						lc.addConnectionCalculator(l, OperationsFactory.conv2DReLU());
					}
				} else
				{
					if (outputActivation != null && l == nn.getOutputLayer())
					{
						ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) OperationsFactory.weightedSum();
						if (l == nn.getOutputLayer() && outputActivation != null)
						{
							cc.addActivationFunction(outputActivation);
						}
						lc.addConnectionCalculator(l, cc);
					} else
					{
						lc.addConnectionCalculator(l, OperationsFactory.reluConnectionCalculator());
					}
				}
			} else
			{
				lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
			}
		}

		return lc;
	}

	public static LayerCalculatorImpl lcTanh(NeuralNetworkImpl nn, TensorFunction outputActivation)
	{
		LayerCalculatorImpl lc = new LayerCalculatorImpl();
		for (Layer l : nn.getLayers())
		{
			if (!Util.isBias(l))
			{
				if (Util.isConvolutional(l))
				{
					if (outputActivation != null && l == nn.getOutputLayer())
					{
						ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) OperationsFactory.conv2D();
						if (l == nn.getOutputLayer() && outputActivation != null)
						{
							cc.addActivationFunction(outputActivation);
						}
						lc.addConnectionCalculator(l, cc);
					} else
					{
						lc.addConnectionCalculator(l, OperationsFactory.conv2DTanh());
					}
				} else
				{
					if (outputActivation != null && l == nn.getOutputLayer())
					{
						ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) OperationsFactory.weightedSum();
						if (l == nn.getOutputLayer() && outputActivation != null)
						{
							cc.addActivationFunction(outputActivation);
						}
						lc.addConnectionCalculator(l, cc);
					} else
					{
						lc.addConnectionCalculator(l, OperationsFactory.tanhConnectionCalculator());
					}
				}
			} else
			{
				lc.addConnectionCalculator(l, new ConstantConnectionCalculator());
			}
		}

		return lc;
	}

	public static void lcMaxPooling(NeuralNetworkImpl nn)
	{
		if (nn.getLayerCalculator() instanceof LayerCalculatorImpl)
		{
			LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
			nn.getLayers().stream().filter(l -> Util.isSubsampling(l)).forEach(l -> lc.addConnectionCalculator(l, OperationsFactory.maxPooling2D()));
		} else
		{
			throw new IllegalArgumentException("LayerCalculator type not supported");
		}
	}

	public static void lcAveragePooling(NeuralNetworkImpl nn)
	{
		if (nn.getLayerCalculator() instanceof LayerCalculatorImpl)
		{
			LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
			nn.getLayers().stream().filter(l -> Util.isSubsampling(l)).forEach(l -> lc.addConnectionCalculator(l, OperationsFactory.averagePooling2D()));
		} else
		{
			throw new IllegalArgumentException("LayerCalculator type not supported");
		}
	}

	public static void lcStochasticPooling(NeuralNetworkImpl nn)
	{
		if (nn.getLayerCalculator() instanceof LayerCalculatorImpl)
		{
			LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
			nn.getLayers().stream().filter(l -> Util.isSubsampling(l)).forEach(l -> lc.addConnectionCalculator(l, OperationsFactory.stochasticPooling2D()));
		} else
		{
			throw new IllegalArgumentException("LayerCalculator type not supported");
		}
	}

	public static void lcDropout(BackPropagationTrainer<?> trainer)
	{
		NeuralNetworkImpl nn = (NeuralNetworkImpl) trainer.getNeuralNetwork();
		if (nn.getLayerCalculator() instanceof LayerCalculatorImpl)
		{
			LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
			nn.getConnections().stream().filter(c -> c instanceof FullyConnected && c.getOutputLayer() != nn.getOutputLayer() && !Util.isBias(c.getInputLayer())).forEach(c -> {
				ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) lc.getConnectionCalculator(c.getOutputLayer());
				Hyperparameters hp = trainer.getHyperparameters();
				if (hp.getDropoutRate(c) > 0)
				{
					cc.addActivationFunction(OperationsFactory.noiseMask(hp.getDropoutRate(c), 0));
				}
			});
		} else
		{
			throw new IllegalArgumentException("LayerCalculator type not supported");
		}
	}

	public static NeuralNetworkImpl mlpSigmoid(int[] layers, boolean addBias)
	{
		NeuralNetworkImpl result = NNFactory.mlp(layers, addBias);
		result.setLayerCalculator(lcSigmoid(result, null));
		return result;
	}

	public static NeuralNetworkImpl mlpSoftRelu(int[] layers, boolean addBias, TensorFunction outputActivation)
	{
		NeuralNetworkImpl result = NNFactory.mlp(layers, addBias);
		result.setLayerCalculator(lcSoftRelu(result, outputActivation));
		return result;
	}

	public static NeuralNetworkImpl mlpRelu(int[] layers, boolean addBias, TensorFunction outputActivation)
	{
		NeuralNetworkImpl result = NNFactory.mlp(layers, addBias);
		result.setLayerCalculator(lcRelu(result, outputActivation));
		return result;
	}

	public static NeuralNetworkImpl mlpTanh(int[] layers, boolean addBias, TensorFunction outputActivation)
	{
		NeuralNetworkImpl result = NNFactory.mlp(layers, addBias);
		result.setLayerCalculator(lcTanh(result, outputActivation));
		return result;
	}

	public static NeuralNetworkImpl maxout(int[] layers, boolean addBias)
	{
		NeuralNetworkImpl result = NNFactory.mlp(layers, addBias);
		result.setLayerCalculator(lcMaxout(result));
		result.getConnections().stream().filter(c -> c instanceof FullyConnected).forEach(c -> {
			MaxoutWinners.getInstance().addConnections(c);
		});

		return result;
	}

	public static Autoencoder autoencoder(int visibleCount, int hiddenCount, boolean addBias)
	{
		Autoencoder result = new Autoencoder();
		NNFactory.mlp(result, new ConnectionFactory(), new int[] { visibleCount, hiddenCount, visibleCount }, addBias);
		return result;
	}

	public static Autoencoder autoencoderSigmoid(int visibleCount, int hiddenCount, boolean addBias)
	{
		Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
		ae.setLayerCalculator(lcSigmoid(ae, null));
		return ae;
	}

	public static Autoencoder autoencoderSoftReLU(int visibleCount, int hiddenCount, boolean addBias, TensorFunction outputActivation)
	{
		Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
		ae.setLayerCalculator(lcSoftRelu(ae, outputActivation));
		return ae;
	}

	public static Autoencoder autoencoderReLU(int visibleCount, int hiddenCount, boolean addBias, TensorFunction outputActivation)
	{
		Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
		ae.setLayerCalculator(lcRelu(ae, outputActivation));
		return ae;
	}

	public static Autoencoder autoencoderTanh(int visibleCount, int hiddenCount, boolean addBias, TensorFunction outputActivation)
	{
		Autoencoder ae = autoencoder(visibleCount, hiddenCount, addBias);
		ae.setLayerCalculator(lcTanh(ae, outputActivation));
		return ae;
	}

	public static DBN dbnSigmoid(int[] layers, boolean addBias)
	{
		DBN result = NNFactory.dbn(layers, addBias);
		result.setLayerCalculator(lcSigmoid(result, null));
		return result;
	}

	public static DBN dbnSoftReLU(int[] layers, boolean addBias)
	{
		DBN result = NNFactory.dbn(layers, addBias);
		result.setLayerCalculator(lcSoftRelu(result, null));
		return result;
	}

	public static DBN dbnReLU(int[] layers, boolean addBias)
	{
		DBN result = NNFactory.dbn(layers, addBias);
		result.setLayerCalculator(lcRelu(result, null));
		return result;
	}

	public static DBN dbnTanh(int[] layers, boolean addBias)
	{
		DBN result = NNFactory.dbn(layers, addBias);
		result.setLayerCalculator(lcTanh(result, null));
		return result;
	}

	public static StackedAutoencoder saeSigmoid(int[] layers, boolean addBias)
	{
		StackedAutoencoder sae = NNFactory.sae(layers, addBias);
		sae.setLayerCalculator(lcSigmoid(sae, null));
		return sae;
	}

	public static StackedAutoencoder saeSoftReLU(int[] layers, boolean addBias)
	{
		StackedAutoencoder sae = NNFactory.sae(layers, addBias);
		sae.setLayerCalculator(lcSoftRelu(sae, null));
		return sae;
	}

	public static StackedAutoencoder saeReLU(int[] layers, boolean addBias)
	{
		StackedAutoencoder sae = NNFactory.sae(layers, addBias);
		sae.setLayerCalculator(lcRelu(sae, null));
		return sae;
	}

	public static StackedAutoencoder saeTanh(int[] layers, boolean addBias)
	{
		StackedAutoencoder sae = NNFactory.sae(layers, addBias);
		sae.setLayerCalculator(lcTanh(sae, null));
		return sae;
	}

	public static RBMLayerCalculator rbmWeightedSumWeightedSum(RBM rbm, int batchSize)
	{
		return new RBMLayerCalculator(rbm, batchSize, OperationsFactory.weightedSum(), OperationsFactory.weightedSum(), OperationsFactory.weightedSum());
	}

	public static RBMLayerCalculator rbmSigmoidSigmoid(RBM rbm, int batchSize)
	{
		return new RBMLayerCalculator(rbm, batchSize, OperationsFactory.sigmoidConnectionCalculator(), OperationsFactory.sigmoidConnectionCalculator(), OperationsFactory.sigmoidConnectionCalculator());
	}

	public static RBMLayerCalculator rbmSoftReluSoftRelu(RBM rbm, int batchSize)
	{
		return new RBMLayerCalculator(rbm, batchSize, OperationsFactory.softReLUConnectionCalculator(), OperationsFactory.softReLUConnectionCalculator(), OperationsFactory.softReLUConnectionCalculator());
	}

	public static RBMLayerCalculator rbmReluRelu(RBM rbm, int batchSize)
	{
		return new RBMLayerCalculator(rbm, batchSize, OperationsFactory.reluConnectionCalculator(), OperationsFactory.reluConnectionCalculator(), OperationsFactory.reluConnectionCalculator());
	}

	public static RBMLayerCalculator rbmTanhTanh(RBM rbm, int batchSize)
	{
		return new RBMLayerCalculator(rbm, batchSize, OperationsFactory.tanhConnectionCalculator(), OperationsFactory.tanhConnectionCalculator(), OperationsFactory.tanhConnectionCalculator());
	}
}
