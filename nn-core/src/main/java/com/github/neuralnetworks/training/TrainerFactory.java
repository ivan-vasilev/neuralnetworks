package com.github.neuralnetworks.training;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.DNN;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.BreadthFirstOrderStrategy;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.calculation.LayerOrderStrategy.ConnectionCandidate;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorTensorFunctions;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.calculation.operations.ClearValuesManager.ClearValuesEventListener;
import com.github.neuralnetworks.calculation.operations.aparapi.rbm.AparapiCDTrainer;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLManagementListener;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationAutoencoder;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationLayerCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.rbm.DBNTrainer;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Util;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

/**
 * Factory for trainers
 */
public class TrainerFactory
{

	/**
	 * Backpropagation trainer Depends on the LayerCalculator of the network
	 */
	public static BackPropagationTrainer<?> backPropagation(NeuralNetworkImpl nn, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand,
			float learningRate, float momentum, float l1weightDecay, float l2weightDecay, float dropoutRate, int trainingBatchSize, int testBatchSize, int epochs)
	{
		Properties p = backpropProperties(nn, trainingSet, testingSet, error, rand, learningRate, momentum, l1weightDecay, l2weightDecay, dropoutRate, trainingBatchSize, testBatchSize, epochs);

		BackPropagationTrainer<NeuralNetwork> result = new BackPropagationTrainer<NeuralNetwork>(p);
		CalculationFactory.lcDropout(result);

		p.setParameter(Constants.BACKPROPAGATION, bplc(nn, p));
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL)
		{
			result.addEventListener(new OpenCLManagementListener(), 0);
		}

		result.addEventListener(new ClearValuesEventListener());

		return result;
	}

	public static BackPropagationLayerCalculatorImpl bplc(NeuralNetworkImpl nn, Properties p)
	{
		BackPropagationLayerCalculatorImpl blc = new BackPropagationLayerCalculatorImpl();
		LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();

		List<ConnectionCandidate> connections = new BreadthFirstOrderStrategy(nn, nn.getOutputLayer()).order();

		connections.forEach(c -> c.target = Util.getOppositeLayer(c.connection, c.target));

		if (connections.size() > 0)
		{
			Layer current = null;
			List<Connections> chunk = new ArrayList<>();
			Set<Layer> convCalculatedLayers = new HashSet<>();

			// the output layer first
			convCalculatedLayers.add(nn.getOutputLayer());

			ConnectionCalculatorTensorFunctions outcc = (ConnectionCalculatorTensorFunctions) lc.getConnectionCalculator(nn.getOutputLayer());
			for (int j = outcc.getActivationFunctions().size() - 1; j >= 0; j--)
			{
				TensorFunction f = outcc.getActivationFunctions().get(j);
				if (OperationsFactory.isSigmoidFunction(f))
				{
					blc.setOutputDerivative(OperationsFactory.sigmoidDerivativeFunction());
				} else if (OperationsFactory.isTanhFunction(f))
				{
					blc.setOutputDerivative(OperationsFactory.tanhDerivativeFunction());
				} else if (OperationsFactory.isSoftReLUFunction(f))
				{
					blc.setOutputDerivative(OperationsFactory.softReLUDerivativeFunction());
				} else if (OperationsFactory.isReLUFunction(f))
				{
					blc.setOutputDerivative(OperationsFactory.reLUDerivativeFunction());
				}
			}

			for (int i = 0; i < connections.size(); i++)
			{
				ConnectionCandidate c = connections.get(i);
				chunk.add(c.connection);

				if (i == connections.size() - 1 || connections.get(i + 1).target != c.target)
				{
					current = c.target;

					ConnectionCalculator result = null;
					ConnectionCalculator ffcc = null;
					if (Util.isBias(current))
					{
						if (current.getConnections().get(0).getOutputLayer() == nn.getOutputLayer())
						{
							ffcc = lc.getConnectionCalculator(nn.getLayers().stream().filter(l -> l != nn.getOutputLayer() && !Util.isSubsampling(l) && !Util.isBias(l)).findFirst().get());
						} else
						{
							ffcc = lc.getConnectionCalculator(current.getConnections().get(0).getOutputLayer());
						}
					} else if (Util.isConvolutional(current) || Util.isSubsampling(current))
					{
						if (chunk.size() != 1)
						{
							throw new IllegalArgumentException("Convolutional layer with more than one connection");
						}

						ffcc = lc.getConnectionCalculator(Util.getOppositeLayer(chunk.iterator().next(), current));
					} else
					{
						ffcc = lc.getConnectionCalculator(current);
					}

					if (OperationsFactory.isMaxout(ffcc))
					{
						result = OperationsFactory.bpMaxout(p);
					} else if (OperationsFactory.isLRNConnectionCalculator(ffcc))
					{
						result = OperationsFactory.bpLRN(p, ffcc);
					} else if (OperationsFactory.isMaxPooling2D(ffcc) || OperationsFactory.isStochasticPooling2D(ffcc))
					{
						result = OperationsFactory.bpMaxPooling(p);
					} else if (OperationsFactory.isAveragePooling2D(ffcc))
					{
						result = OperationsFactory.bpAveragePooling(p);
					} else {
						result = OperationsFactory.bpConnectionCalculator(c.connection, p);
					}

					if (result != null)
					{
						blc.addConnectionCalculator(current, result);
					}

					chunk.clear();
				}
			}

			nn.getLayers().stream().filter(l -> !Util.isBias(l) && nn.getOutputLayer() != l && lc.getConnectionCalculator(l) instanceof ConnectionCalculatorTensorFunctions).forEach(l -> {
				ConnectionCalculatorTensorFunctions cc = (ConnectionCalculatorTensorFunctions) lc.getConnectionCalculator(l);
				BackPropagationConnectionCalculatorImpl bcc = (BackPropagationConnectionCalculatorImpl) blc.getConnectionCalculator(l);

				for (int j = cc.getActivationFunctions().size() - 1; j >= 0; j--)
				{
					TensorFunction f = cc.getActivationFunctions().get(j);
					if (OperationsFactory.isSigmoidFunction(f))
					{
						bcc.addActivationFunction(OperationsFactory.sigmoidDerivativeFunction());
					} else if (OperationsFactory.isTanhFunction(f))
					{
						bcc.addActivationFunction(OperationsFactory.tanhDerivativeFunction());
					} else if (OperationsFactory.isSoftReLUFunction(f))
					{
						bcc.addActivationFunction(OperationsFactory.softReLUDerivativeFunction());
					} else if (OperationsFactory.isReLUFunction(f))
					{
						bcc.addActivationFunction(OperationsFactory.reLUDerivativeFunction());
					} else if (OperationsFactory.isNoiseMask(f))
					{
						bcc.addActivationFunction(OperationsFactory.mask(f));
					}
				}
			});
		}

		return blc;
	}

	public static BackPropagationAutoencoder backPropagationAutoencoder(NeuralNetworkImpl nn, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error,
			NNRandomInitializer rand, float learningRate, float momentum, float l1weightDecay, float l2weightDecay, float inputCorruptionRate, int trainingBatchSize, int testBatchSize, int epochs)
	{
		Properties p = backpropProperties(nn, trainingSet, testingSet, error, rand, learningRate, momentum, l1weightDecay, l2weightDecay, 0F, trainingBatchSize, testBatchSize, epochs);
		p.setParameter(Constants.CORRUPTION_LEVEL, inputCorruptionRate);
		p.setParameter(Constants.BACKPROPAGATION, bplc(nn, p));

		return new BackPropagationAutoencoder(p);
	}

	protected static Properties backpropProperties(NeuralNetworkImpl nn, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand,
			float learningRate, float momentum, float l1weightDecay, float l2weightDecay, float dropoutRate, int trainingBatchSize, int testBatchSize, int epochs)
	{
		Properties p = new Properties();
		p.setParameter(Constants.NEURAL_NETWORK, nn);
		p.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingSet);
		p.setParameter(Constants.TESTING_INPUT_PROVIDER, testingSet);

		Hyperparameters hp = new Hyperparameters();
		hp.setDefaultLearningRate(learningRate);
		hp.setDefaultMomentum(momentum);
		hp.setDefaultL1WeightDecay(l1weightDecay);
		hp.setDefaultL2WeightDecay(l2weightDecay);
		hp.setDefaultDropoutRate(dropoutRate);

		p.setParameter(Constants.HYPERPARAMETERS, hp);
		p.setParameter(Constants.WEIGHT_UDPATES, TensorFactory.duplicate(nn.getProperties().getParameter(Constants.WEIGHTS_PROVIDER)));
		p.setParameter(Constants.OUTPUT_ERROR, error);
		p.setParameter(Constants.RANDOM_INITIALIZER, rand);
		p.setParameter(Constants.TRAINING_BATCH_SIZE, trainingBatchSize);
		p.setParameter(Constants.TEST_BATCH_SIZE, testBatchSize);
		p.setParameter(Constants.EPOCHS, epochs);

		LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
		if (OperationsFactory.isSoftmaxCC(lc.getConnectionCalculator(nn.getOutputLayer())))
		{
			p.setParameter(Constants.LOSS_FUNCTION, OperationsFactory.softmaxLoss());
		} else
		{
			p.setParameter(Constants.LOSS_FUNCTION, OperationsFactory.mse());
		}

		return p;
	}

	public static AparapiCDTrainer cdSoftReLUTrainer(RBM rbm, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand, float learningRate,
			float momentum, float l1weightDecay, float l2weightDecay, int gibbsSampling, int trainingBatchSize, int epochs, boolean isPersistentCD)
	{
		rbm.setLayerCalculator(CalculationFactory.lcSoftRelu(rbm, null));

		RBMLayerCalculator lc = CalculationFactory.rbmSoftReluSoftRelu(rbm, trainingBatchSize);
		ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) lc.getNegPhaseHiddenToVisibleCC();
		cc.addInputModifierFunction(OperationsFactory.bernoulliDistribution());

		return new AparapiCDTrainer(rbmProperties(rbm, lc, trainingSet, testingSet, error, rand, learningRate, momentum, l1weightDecay, l2weightDecay, gibbsSampling, trainingBatchSize, epochs,
				isPersistentCD));
	}

	public static AparapiCDTrainer cdSigmoidBinaryTrainer(RBM rbm, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand, float learningRate,
			float momentum, float l1weightDecay, float l2weightDecay, int gibbsSampling, int trainingBatchSize, int epochs, boolean isPersistentCD)
	{
		rbm.setLayerCalculator(CalculationFactory.lcSigmoid(rbm, null));

		RBMLayerCalculator lc = CalculationFactory.rbmSigmoidSigmoid(rbm, trainingBatchSize);
		ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) lc.getNegPhaseHiddenToVisibleCC();
		cc.addInputModifierFunction(OperationsFactory.bernoulliDistribution());

		return new AparapiCDTrainer(rbmProperties(rbm, lc, trainingSet, testingSet, error, rand, learningRate, momentum, l1weightDecay, l2weightDecay, gibbsSampling, trainingBatchSize, epochs,
				isPersistentCD));
	}

	public static AparapiCDTrainer cdSigmoidTrainer(RBM rbm, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand, float learningRate,
			float momentum, float l1weightDecay, float l2weightDecay, int gibbsSampling, int trainingBatchSize, int epochs, boolean isPersistentCD)
	{
		rbm.setLayerCalculator(CalculationFactory.lcSigmoid(rbm, null));
		RBMLayerCalculator lc = CalculationFactory.rbmSigmoidSigmoid(rbm, trainingBatchSize);
		return new AparapiCDTrainer(rbmProperties(rbm, lc, trainingSet, testingSet, error, rand, learningRate, momentum, l1weightDecay, l2weightDecay, gibbsSampling, trainingBatchSize, epochs,
				isPersistentCD));
	}

	protected static Properties rbmProperties(RBM rbm, RBMLayerCalculator lc, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand,
			float learningRate, float momentum, float l1weightDecay, float l2weightDecay, int gibbsSampling, int trainingBatchSize, int epochs, boolean isPersistentCD)
	{
		Properties p = new Properties();
		p.setParameter(Constants.NEURAL_NETWORK, rbm);
		p.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingSet);
		p.setParameter(Constants.TESTING_INPUT_PROVIDER, testingSet);

		Hyperparameters hp = new Hyperparameters();
		hp.setDefaultLearningRate(learningRate);
		hp.setDefaultMomentum(momentum);
		hp.setDefaultL1WeightDecay(l1weightDecay);
		hp.setDefaultL2WeightDecay(l2weightDecay);
		hp.setDefault(Constants.GIBBS_SAMPLING_COUNT, gibbsSampling);
		p.setParameter(Constants.HYPERPARAMETERS, hp);

		p.setParameter(Constants.OUTPUT_ERROR, error);
		p.setParameter(Constants.RANDOM_INITIALIZER, rand);
		p.setParameter(Constants.PERSISTENT_CD, isPersistentCD);
		p.setParameter(Constants.LAYER_CALCULATOR, lc);
		p.setParameter(Constants.TRAINING_BATCH_SIZE, trainingBatchSize);
		p.setParameter(Constants.EPOCHS, epochs);

		return p;
	}

	public static DNNLayerTrainer dnnLayerTrainer(DNN<?> dnn, Map<NeuralNetwork, OneStepTrainer<?>> layerTrainers, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error)
	{
		return new DNNLayerTrainer(layerTrainerProperties(dnn, layerTrainers, trainingSet, testingSet, error));
	}

	public static DBNTrainer dbnTrainer(DNN<?> dnn, Map<NeuralNetwork, OneStepTrainer<?>> layerTrainers, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error)
	{
		return new DBNTrainer(layerTrainerProperties(dnn, layerTrainers, trainingSet, testingSet, error));
	}

	protected static Properties layerTrainerProperties(DNN<?> dnn, Map<NeuralNetwork, OneStepTrainer<?>> layerTrainers, TrainingInputProvider trainingSet, TrainingInputProvider testingSet,
			OutputError error)
	{
		Properties p = new Properties();
		p.setParameter(Constants.NEURAL_NETWORK, dnn);
		p.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingSet);
		p.setParameter(Constants.TESTING_INPUT_PROVIDER, testingSet);
		p.setParameter(Constants.OUTPUT_ERROR, error);
		p.setParameter(Constants.LAYER_TRAINERS, layerTrainers);

		return p;
	}
}
