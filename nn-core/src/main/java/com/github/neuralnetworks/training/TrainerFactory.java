package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.architecture.types.SupervisedRBM;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.calculation.SupervisedRBMLayerCalculator;
import com.github.neuralnetworks.training.rbm.CDAparapiTrainer;
import com.github.neuralnetworks.training.rbm.PCDAparapiTrainer;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

public class TrainerFactory {

    public static BackPropagationTrainer backPropagation(NeuralNetwork nn, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, RandomInitializer rand, float learningRate, float momentum, float weightDecay) {
	Properties p = new Properties();
	p.setParameter(Constants.NEURAL_NETWORK, nn);
	p.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingSet);
	p.setParameter(Constants.TESTING_INPUT_PROVIDER, testingSet);
	p.setParameter(Constants.LAYER_CALCULATOR, new LayerCalculatorImpl());
	p.setParameter(Constants.LEARNING_RATE, learningRate);
	p.setParameter(Constants.MOMENTUM, momentum);
	p.setParameter(Constants.WEIGHT_DECAY, weightDecay);
	p.setParameter(Constants.OUTPUT_ERROR, error);
	p.setParameter(Constants.RANDOM_INITIALIZER, rand);

	BackPropagationTrainer result = new BackPropagationTrainer(p);

	return result;
    }

    public static CDAparapiTrainer cdTrainer(RBM rbm, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, RandomInitializer rand, float learningRate, float momentum, float weightDecay, int gibbsSampling) {
	return new CDAparapiTrainer(rbmProperties(rbm, trainingSet, testingSet, error, rand, learningRate, momentum, weightDecay, gibbsSampling));
    }

    public static PCDAparapiTrainer pcdTrainer(RBM rbm, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, RandomInitializer rand, float learningRate, float momentum, float weightDecay, int gibbsSampling) {
	return new PCDAparapiTrainer(rbmProperties(rbm, trainingSet, testingSet, error, rand, learningRate, momentum, weightDecay, gibbsSampling));
    }

    protected static Properties rbmProperties(RBM rbm, TrainingInputProvider trainingSet, TrainingInputProvider testingSet, OutputError error, RandomInitializer rand, float learningRate, float momentum, float weightDecay, int gibbsSampling) {
	Properties p = new Properties();
	p.setParameter(Constants.NEURAL_NETWORK, rbm);
	p.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingSet);
	p.setParameter(Constants.TESTING_INPUT_PROVIDER, testingSet);
	p.setParameter(Constants.LAYER_CALCULATOR, rbm instanceof SupervisedRBM ? new SupervisedRBMLayerCalculator((SupervisedRBM) rbm) : new RBMLayerCalculator(rbm));
	p.setParameter(Constants.LEARNING_RATE, learningRate);
	p.setParameter(Constants.MOMENTUM, momentum);
	p.setParameter(Constants.WEIGHT_DECAY, weightDecay);
	p.setParameter(Constants.GIBBS_SAMPLING_COUNT, gibbsSampling);
	p.setParameter(Constants.OUTPUT_ERROR, error);
	p.setParameter(Constants.RANDOM_INITIALIZER, rand);

	return p;
    }
}
