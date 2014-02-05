package com.github.neuralnetworks.training;

import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.DNN;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiAveragePooling2D;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DSoftReLU;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiConv2DTanh;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiMaxPooling2D;
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
import com.github.neuralnetworks.training.backpropagation.MSEDerivative;
import com.github.neuralnetworks.training.random.RandomInitializer;
import com.github.neuralnetworks.training.rbm.AparapiCDTrainer;
import com.github.neuralnetworks.training.rbm.DBNTrainer;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Util;

/**
 * Factory for trainers
 */
public class TrainerFactory {

    /**
     * Backpropagation trainer
     * Depends on the LayerCalculator of the network
     * 
     * @param nn
     * @param trainingSet
     * @param testingSet
     * @param error
     * @param rand
     * @param learningRate
     * @param momentum
     * @param weightDecay
     * @return
     */
    public static BackPropagationTrainer<?> backPropagation(NeuralNetworkImpl nn, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, RandomInitializer rand, float learningRate, float momentum, float weightDecay) {
	BackPropagationTrainer<?> t = new BackPropagationTrainer<NeuralNetwork>(backpropProperties(nn, trainingSet, testingSet, error, rand, learningRate, momentum, weightDecay));

	BackPropagationLayerCalculatorImpl bplc = bplc(nn, t.getProperties());
	t.getProperties().setParameter(Constants.BACKPROPAGATION, bplc);

	return t;
    }

    private static BackPropagationLayerCalculatorImpl bplc(NeuralNetworkImpl nn, Properties p) {
	BackPropagationLayerCalculatorImpl blc = new BackPropagationLayerCalculatorImpl();

	for (Layer l : nn.getLayers()) {
	    if (l != nn.getOutputLayer()) {
		ConnectionCalculator cc = createCC(nn, l, p);
		if (cc != null) {
		    blc.addConnectionCalculator(l, cc);
		}
	    }
	}

	return blc;
    }

    private static ConnectionCalculator createCC(NeuralNetwork nn, Layer l, Properties p) {
	ConnectionCalculator result = null;

	LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
	ConnectionCalculator cc = Util.isBias(l) ? lc.getConnectionCalculator(l.getConnections().get(0).getOutputLayer()) : lc.getConnectionCalculator(l);

	if (cc instanceof AparapiSigmoid) {
	    result = new BackPropagationSigmoid(p);
	} else if (cc instanceof AparapiTanh) {
	    result = new BackPropagationTanh(p);
	} else if (cc instanceof AparapiSoftReLU) {
	    result = new BackPropagationSoftReLU(p);
	} else if (cc instanceof AparapiReLU) {
	    result = new BackPropagationReLU(p);
	} else if (cc instanceof AparapiMaxPooling2D || cc instanceof AparapiStochasticPooling2D) {
	    result = new BackpropagationMaxPooling2D();
	} else if (cc instanceof AparapiAveragePooling2D) {
	    result = new BackpropagationAveragePooling2D();
	} else if (cc instanceof ConnectionCalculatorConv) {
	    boolean hasFullyConnected = false;
	    boolean hasOutputConnection = false;
	    for (Connections c : l.getConnections()) {
		if (c instanceof FullyConnected && c.getInputLayer() == l) {
		    hasFullyConnected = true;
		}

		if (Util.getOppositeLayer(c, l) == nn.getOutputLayer()) {
		    hasOutputConnection = true;
		}
	    }

	    if (cc instanceof AparapiConv2DSigmoid) {
		result = hasFullyConnected ? new BackPropagationSigmoid(p) : hasOutputConnection ? new BackPropagationConv2D(p) : new BackPropagationConv2DSigmoid(p);
	    } else if (cc instanceof AparapiConv2DTanh) {
		result = hasFullyConnected ? new BackPropagationTanh(p) : hasOutputConnection ? new BackPropagationConv2D(p) : new BackPropagationConv2DTanh(p);
	    } else if (cc instanceof AparapiConv2DSoftReLU) {
		result = hasFullyConnected ? new BackPropagationSoftReLU(p) : hasOutputConnection ? new BackPropagationConv2D(p) : new BackPropagationConv2DSoftReLU(p);
	    } else if (cc instanceof AparapiConv2DReLU) {
		result = hasFullyConnected ? new BackPropagationReLU(p) : hasOutputConnection ? new BackPropagationConv2D(p) : new BackPropagationConv2DReLU(p);
	    }
	}

	return result;
    }

    public static BackPropagationAutoencoder backPropagationAutoencoder(NeuralNetworkImpl nn, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, RandomInitializer rand, float learningRate, float momentum, float weightDecay, float inputCorruptionRate) {
	BackPropagationAutoencoder t = new BackPropagationAutoencoder(backpropProperties(nn, trainingSet, testingSet, error, rand, learningRate, momentum, weightDecay));

	BackPropagationLayerCalculatorImpl bplc = bplc(nn, t.getProperties());
	t.getProperties().setParameter(Constants.BACKPROPAGATION, bplc);

	return t;
    }

    protected static Properties backpropProperties(NeuralNetwork nn, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, RandomInitializer rand, float learningRate, float momentum, float weightDecay) {
	Properties p = new Properties();
	p.setParameter(Constants.NEURAL_NETWORK, nn);
	p.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingSet);
	p.setParameter(Constants.TESTING_INPUT_PROVIDER, testingSet);
	p.setParameter(Constants.LEARNING_RATE, learningRate);
	p.setParameter(Constants.MOMENTUM, momentum);
	p.setParameter(Constants.WEIGHT_DECAY, weightDecay);
	p.setParameter(Constants.LAYER_CALCULATOR, nn.getLayerCalculator());
	p.setParameter(Constants.OUTPUT_ERROR_DERIVATIVE, new MSEDerivative());
	p.setParameter(Constants.OUTPUT_ERROR, error);
	p.setParameter(Constants.RANDOM_INITIALIZER, rand);

	return p;
    }

    public static AparapiCDTrainer cdSoftReLUTrainer(RBM rbm, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, RandomInitializer rand, float learningRate, float momentum, float weightDecay, int gibbsSampling, boolean isPersistentCD) {
	rbm.setLayerCalculator(NNFactory.rbmSoftReluSoftRelu(rbm));

	RBMLayerCalculator lc = NNFactory.rbmSigmoidSigmoid(rbm);
	ConnectionCalculatorFullyConnected cc = (ConnectionCalculatorFullyConnected) lc.getConnectionCalculator(rbm.getInputLayer());
	cc.addPreTransferFunction(new BernoulliDistribution());
	
	return new AparapiCDTrainer(rbmProperties(rbm, lc, trainingSet, testingSet, error, rand, learningRate, momentum, weightDecay, gibbsSampling, isPersistentCD));
    }

    public static AparapiCDTrainer cdSigmoidTrainer(RBM rbm, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, RandomInitializer rand, float learningRate, float momentum, float weightDecay, int gibbsSampling, boolean isPersistentCD) {
	rbm.setLayerCalculator(NNFactory.rbmSigmoidSigmoid(rbm));

	RBMLayerCalculator lc = NNFactory.rbmSigmoidSigmoid(rbm);
	ConnectionCalculatorFullyConnected cc = (ConnectionCalculatorFullyConnected) lc.getConnectionCalculator(rbm.getInputLayer());
	cc.addPreTransferFunction(new BernoulliDistribution());

	return new AparapiCDTrainer(rbmProperties(rbm, lc, trainingSet, testingSet, error, rand, learningRate, momentum, weightDecay, gibbsSampling, isPersistentCD));
    }

    protected static Properties rbmProperties(RBM rbm, RBMLayerCalculator lc, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, RandomInitializer rand, float learningRate, float momentum, float weightDecay, int gibbsSampling, boolean resetRBM) {
	Properties p = new Properties();
	p.setParameter(Constants.NEURAL_NETWORK, rbm);
	p.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingSet);
	p.setParameter(Constants.TESTING_INPUT_PROVIDER, testingSet);
	p.setParameter(Constants.LAYER_CALCULATOR, new RBMLayerCalculator());
	p.setParameter(Constants.LEARNING_RATE, learningRate);
	p.setParameter(Constants.MOMENTUM, momentum);
	p.setParameter(Constants.WEIGHT_DECAY, weightDecay);
	p.setParameter(Constants.GIBBS_SAMPLING_COUNT, gibbsSampling);
	p.setParameter(Constants.OUTPUT_ERROR, error);
	p.setParameter(Constants.RANDOM_INITIALIZER, rand);
	p.setParameter(Constants.RESET_RBM, resetRBM);
	p.setParameter(Constants.LAYER_CALCULATOR, lc);

	return p;
    }

    public static DNNLayerTrainer dnnLayerTrainer(DNN<?> dnn, Map<NeuralNetwork, OneStepTrainer<?>> layerTrainers, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error) {
	return new DNNLayerTrainer(layerTrainerProperties(dnn, layerTrainers, trainingSet, testingSet, error));
    }

    public static DBNTrainer dbnTrainer(DNN<?> dnn, Map<NeuralNetwork, OneStepTrainer<?>> layerTrainers, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error) {
	return new DBNTrainer(layerTrainerProperties(dnn, layerTrainers, trainingSet, testingSet, error));
    }

    protected static Properties layerTrainerProperties(DNN<?> dnn, Map<NeuralNetwork, OneStepTrainer<?>> layerTrainers, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error) {
	Properties p = new Properties();
	p.setParameter(Constants.NEURAL_NETWORK, dnn);
	p.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingSet);
	p.setParameter(Constants.TESTING_INPUT_PROVIDER, testingSet);
	p.setParameter(Constants.LAYER_CALCULATOR, new RBMLayerCalculator());
	p.setParameter(Constants.OUTPUT_ERROR, error);
	p.setParameter(Constants.LAYER_TRAINERS, layerTrainers);

	return p;
    }
}
