package com.github.neuralnetworks.training;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.WeightsConnections;
import com.github.neuralnetworks.architecture.types.DNN;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.BreadthFirstOrderStrategy;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.LayerOrderStrategy.ConnectionCandidate;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiAveragePooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DSoftReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DTanh;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiMaxPooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiMaxout;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSoftReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiStochasticPooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiTanh;
import com.github.neuralnetworks.calculation.neuronfunctions.BernoulliDistribution;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorConv;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorFullyConnected;
import com.github.neuralnetworks.training.backpropagation.BackPropagationAutoencoder;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConv2D;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConv2DReLU;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConv2DSigmoid;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConv2DSoftReLU;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConv2DTanh;
import com.github.neuralnetworks.training.backpropagation.BackPropagationLayerCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationReLU;
import com.github.neuralnetworks.training.backpropagation.BackPropagationSigmoid;
import com.github.neuralnetworks.training.backpropagation.BackPropagationSoftReLU;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTanh;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.backpropagation.BackpropagationAveragePooling2D;
import com.github.neuralnetworks.training.backpropagation.BackpropagationMaxPooling2D;
import com.github.neuralnetworks.training.backpropagation.BackpropagationMaxout;
import com.github.neuralnetworks.training.backpropagation.MSEDerivative;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.rbm.AparapiCDTrainer;
import com.github.neuralnetworks.training.rbm.DBNTrainer;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.TensorFactory;
import com.github.neuralnetworks.util.Util;

/**
 * Factory for trainers
 */
public class TrainerFactory {

    /**
     * Backpropagation trainer Depends on the LayerCalculator of the network
     * 
     * @param nn
     * @param trainingSet
     * @param testingSet
     * @param error
     * @param rand
     * @param learningRate
     * @param momentum
     * @param l1weightDecay
     * @return
     */
    public static BackPropagationTrainer<?> backPropagation(NeuralNetworkImpl nn, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand, float learningRate, float momentum, float l1weightDecay, float l2weightDecay, float dropoutRate, int trainingBatchSize, int testBatchSize, int epochs) {
	Properties p = backpropProperties(nn, trainingSet, testingSet, error, rand, learningRate, momentum, l1weightDecay, l2weightDecay, dropoutRate, trainingBatchSize, testBatchSize, epochs);
	p.setParameter(Constants.BACKPROPAGATION, bplc(nn, p));

	return new BackPropagationTrainer<NeuralNetwork>(p);
    }

    private static BackPropagationLayerCalculatorImpl bplc(NeuralNetworkImpl nn, Properties p) {
	BackPropagationLayerCalculatorImpl blc = new BackPropagationLayerCalculatorImpl();
	LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();

	List<ConnectionCandidate> connections = new BreadthFirstOrderStrategy(nn, nn.getOutputLayer()).order();

	if (connections.size() > 0) {
	    Layer current = null;
	    List<Connections> chunk = new ArrayList<>();
	    Set<Layer> convCalculatedLayers = new HashSet<>(); // tracks
							       // convolutional
							       // layers
							       // (because their
							       // calculations
							       // are
							       // interlinked)
	    convCalculatedLayers.add(nn.getOutputLayer());

	    for (int i = 0; i < connections.size(); i++) {
		ConnectionCandidate c = connections.get(i);
		chunk.add(c.connection);

		if (i == connections.size() - 1 || connections.get(i + 1).target != c.target) {
		    current = c.target;

		    ConnectionCalculator result = null;
		    ConnectionCalculator ffcc = null;
		    if (Util.isBias(current)) {
			ffcc = lc.getConnectionCalculator(current.getConnections().get(0).getOutputLayer());
		    } else if (Util.isConvolutional(current) || Util.isSubsampling(current)) {
			if (chunk.size() != 1) {
			    throw new IllegalArgumentException("Convolutional layer with more than one connection");
			}

			ffcc = lc.getConnectionCalculator(Util.getOppositeLayer(chunk.iterator().next(), current));
		    } else {
			ffcc = lc.getConnectionCalculator(current);
		    }

		    if (ffcc instanceof AparapiSigmoid) {
			result = new BackPropagationSigmoid(p);
		    } else if (ffcc instanceof AparapiTanh) {
			result = new BackPropagationTanh(p);
		    } else if (ffcc instanceof AparapiSoftReLU) {
			result = new BackPropagationSoftReLU(p);
		    } else if (ffcc instanceof AparapiReLU) {
			result = new BackPropagationReLU(p);
		    } else if (ffcc instanceof AparapiMaxout) {
			result = new BackpropagationMaxout(p);
		    } else if (ffcc instanceof AparapiMaxPooling2D || ffcc instanceof AparapiStochasticPooling2D) {
			result = new BackpropagationMaxPooling2D();
		    } else if (ffcc instanceof AparapiAveragePooling2D) {
			result = new BackpropagationAveragePooling2D();
		    } else if (ffcc instanceof ConnectionCalculatorConv) {
			Layer opposite = Util.getOppositeLayer(chunk.iterator().next(), current);
			if (!convCalculatedLayers.contains(opposite)) {
			    convCalculatedLayers.add(opposite);

			    if (ffcc instanceof AparapiConv2DSigmoid) {
				result = new BackPropagationConv2DSigmoid(p);
			    } else if (ffcc instanceof AparapiConv2DTanh) {
				result = new BackPropagationConv2DTanh(p);
			    } else if (ffcc instanceof AparapiConv2DSoftReLU) {
				result = new BackPropagationConv2DSoftReLU(p);
			    } else if (ffcc instanceof AparapiConv2DReLU) {
				result = new BackPropagationConv2DReLU(p);
			    }
			} else {
			    result = new BackPropagationConv2D(p);
			}
		    }

		    if (result != null) {
			blc.addConnectionCalculator(current, result);
		    }

		    chunk.clear();
		}
	    }
	}

	return blc;
    }

    public static BackPropagationAutoencoder backPropagationAutoencoder(NeuralNetworkImpl nn, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand, float learningRate, float momentum, float l1weightDecay, float l2weightDecay, float inputCorruptionRate, int trainingBatchSize, int testBatchSize, int epochs) {
	Properties p = backpropProperties(nn, trainingSet, testingSet, error, rand, learningRate, momentum, l1weightDecay, l2weightDecay, 0F, trainingBatchSize, testBatchSize, epochs);
	p.setParameter(Constants.CORRUPTION_LEVEL, inputCorruptionRate);
	p.setParameter(Constants.BACKPROPAGATION, bplc(nn, p));

	return new BackPropagationAutoencoder(p);
    }

    protected static Properties backpropProperties(NeuralNetworkImpl nn, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand, float learningRate, float momentum, float l1weightDecay, float l2weightDecay, float dropoutRate, int trainingBatchSize, int testBatchSize, int epochs) {
	Properties p = new Properties();
	p.setParameter(Constants.NEURAL_NETWORK, nn);
	p.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingSet);
	p.setParameter(Constants.TESTING_INPUT_PROVIDER, testingSet);
	p.setParameter(Constants.LEARNING_RATE, learningRate);
	p.setParameter(Constants.MOMENTUM, momentum);
	p.setParameter(Constants.L1_WEIGHT_DECAY, l1weightDecay);
	p.setParameter(Constants.L2_WEIGHT_DECAY, l2weightDecay);
	p.setParameter(Constants.OUTPUT_ERROR_DERIVATIVE, new MSEDerivative());
	p.setParameter(Constants.WEIGHT_UDPATES, weightUpdates(nn));
	p.setParameter(Constants.OUTPUT_ERROR, error);
	p.setParameter(Constants.RANDOM_INITIALIZER, rand);
	p.setParameter(Constants.TRAINING_BATCH_SIZE, trainingBatchSize);
	p.setParameter(Constants.TEST_BATCH_SIZE, testBatchSize);
	p.setParameter(Constants.EPOCHS, epochs);
	p.setParameter(Constants.DROPOUT_RATE, dropoutRate);

	return p;
    }

    public static AparapiCDTrainer cdSoftReLUTrainer(RBM rbm, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand, float learningRate, float momentum, float l1weightDecay, float l2weightDecay, int gibbsSampling, int trainingBatchSize, int epochs, boolean isPersistentCD) {
	rbm.setLayerCalculator(NNFactory.lcSoftRelu(rbm, null));

	RBMLayerCalculator lc = NNFactory.rbmSoftReluSoftRelu(rbm, trainingBatchSize);
	ConnectionCalculatorFullyConnected cc = (ConnectionCalculatorFullyConnected) lc.getNegPhaseHiddenToVisibleCC();
	cc.addPreTransferFunction(new BernoulliDistribution());

	return new AparapiCDTrainer(rbmProperties(rbm, lc, trainingSet, testingSet, error, rand, learningRate, momentum, l1weightDecay, l2weightDecay, gibbsSampling, trainingBatchSize, epochs, isPersistentCD));
    }

    public static AparapiCDTrainer cdSigmoidBinaryTrainer(RBM rbm, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand, float learningRate, float momentum, float l1weightDecay, float l2weightDecay, int gibbsSampling, int trainingBatchSize, int epochs, boolean isPersistentCD) {
	rbm.setLayerCalculator(NNFactory.lcSigmoid(rbm, null));

	RBMLayerCalculator lc = NNFactory.rbmSigmoidSigmoid(rbm, trainingBatchSize);
	ConnectionCalculatorFullyConnected cc = (ConnectionCalculatorFullyConnected) lc.getNegPhaseHiddenToVisibleCC();
	cc.addPreTransferFunction(new BernoulliDistribution());

	return new AparapiCDTrainer(rbmProperties(rbm, lc, trainingSet, testingSet, error, rand, learningRate, momentum, l1weightDecay, l2weightDecay, gibbsSampling, trainingBatchSize, epochs, isPersistentCD));
    }
    
    public static AparapiCDTrainer cdSigmoidTrainer(RBM rbm, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand, float learningRate, float momentum, float l1weightDecay, float l2weightDecay, int gibbsSampling, int trainingBatchSize, int epochs, boolean isPersistentCD) {
	rbm.setLayerCalculator(NNFactory.lcSigmoid(rbm, null));
	RBMLayerCalculator lc = NNFactory.rbmSigmoidSigmoid(rbm, trainingBatchSize);
	return new AparapiCDTrainer(rbmProperties(rbm, lc, trainingSet, testingSet, error, rand, learningRate, momentum, l1weightDecay, l2weightDecay, gibbsSampling, trainingBatchSize, epochs, isPersistentCD));
    }

    protected static Properties rbmProperties(RBM rbm, RBMLayerCalculator lc, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, NNRandomInitializer rand, float learningRate, float momentum, float l1weightDecay, float l2weightDecay, int gibbsSampling, int trainingBatchSize, int epochs, boolean isPersistentCD) {
	Properties p = new Properties();
	p.setParameter(Constants.NEURAL_NETWORK, rbm);
	p.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingSet);
	p.setParameter(Constants.TESTING_INPUT_PROVIDER, testingSet);
	p.setParameter(Constants.LEARNING_RATE, learningRate);
	p.setParameter(Constants.MOMENTUM, momentum);
	p.setParameter(Constants.L1_WEIGHT_DECAY, l1weightDecay);
	p.setParameter(Constants.L2_WEIGHT_DECAY, l2weightDecay);
	p.setParameter(Constants.GIBBS_SAMPLING_COUNT, gibbsSampling);
	p.setParameter(Constants.OUTPUT_ERROR, error);
	p.setParameter(Constants.RANDOM_INITIALIZER, rand);
	p.setParameter(Constants.PERSISTENT_CD, isPersistentCD);
	p.setParameter(Constants.LAYER_CALCULATOR, lc);
	p.setParameter(Constants.TRAINING_BATCH_SIZE, trainingBatchSize);
	p.setParameter(Constants.EPOCHS, epochs);

	return p;
    }

    public static DNNLayerTrainer dnnLayerTrainer(DNN<?> dnn, Map<NeuralNetwork, OneStepTrainer<?>> layerTrainers, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error) {
	return new DNNLayerTrainer(layerTrainerProperties(dnn, layerTrainers, trainingSet, testingSet, error));
    }

    public static DBNTrainer dbnTrainer(DNN<?> dnn, Map<NeuralNetwork, OneStepTrainer<?>> layerTrainers, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error) {
	return new DBNTrainer(layerTrainerProperties(dnn, layerTrainers, trainingSet, testingSet, error));
    }

    /**
     * @param nn
     * @return Weight update tensors
     */
    public static Map<Connections, Tensor> weightUpdates(NeuralNetworkImpl nn) {
	Map<Connections, Tensor> result = new HashMap<>();

	ConnectionFactory cf = nn.getProperties().getParameter(Constants.CONNECTION_FACTORY);

	List<Connections> connections = cf.getConnections().stream().filter(c -> c instanceof WeightsConnections).collect(Collectors.toList());
	float[] elements = cf.useSharedWeights() ? new float[((WeightsConnections) connections.get(0)).getWeights().getElements().length] : null;
	IntStream.range(0, connections.size()).forEach(i -> result.put(connections.get(i), TensorFactory.duplicate(((WeightsConnections) connections.get(i)).getWeights(), elements)));

	return result;
    }

    protected static Properties layerTrainerProperties(DNN<?> dnn, Map<NeuralNetwork, OneStepTrainer<?>> layerTrainers, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error) {
	Properties p = new Properties();
	p.setParameter(Constants.NEURAL_NETWORK, dnn);
	p.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingSet);
	p.setParameter(Constants.TESTING_INPUT_PROVIDER, testingSet);
	p.setParameter(Constants.OUTPUT_ERROR, error);
	p.setParameter(Constants.LAYER_TRAINERS, layerTrainers);

	return p;
    }
}
