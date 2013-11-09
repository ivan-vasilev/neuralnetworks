package com.github.neuralnetworks.training.rbm;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiSigmoid;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiStochasticBinary;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;

public abstract class CDAparapiTrainerBase extends CDTrainerBase {

    private CDWeightUpdatesKernel weightUpdatesKernel;
    private CDBiasUpdatesKernel visibleBiasUpdatesKernel;
    private CDBiasUpdatesKernel hiddenBiasUpdatesKernel;

    public CDAparapiTrainerBase(Properties properties) {
	super(properties);
	init();
    }

    @Override
    protected void updateWeights(Matrix posPhaseVisible, Matrix posPhaseHidden, Matrix negPhaseVisible, Matrix negPhaseHidden) {
	RBM rbm = getNeuralNetwork();

	weightUpdatesKernel.setPosPhaseVisible(posPhaseVisible.getElements());
	weightUpdatesKernel.setPosPhaseHidden(posPhaseHidden.getElements());
	weightUpdatesKernel.setNegPhaseVisible(negPhaseVisible.getElements());
	weightUpdatesKernel.setNegPhaseHidden(negPhaseHidden.getElements());
	weightUpdatesKernel.setLearningRate(getLearningRate());
	weightUpdatesKernel.setMomentum(getMomentum());
	weightUpdatesKernel.setWeightDecay(getWeightDecay());
	weightUpdatesKernel.setMiniBatchSize(posPhaseVisible.getColumns());
	weightUpdatesKernel.setExecutionMode(Environment.getInstance().getExecutionMode());
	weightUpdatesKernel.execute(rbm.getMainConnections().getConnectionGraph().getElements().length);

	// update visible bias
	if (visibleBiasUpdatesKernel != null) {
	    visibleBiasUpdatesKernel.setPosPhase(posPhaseVisible.getElements());
	    visibleBiasUpdatesKernel.setNegPhase(negPhaseVisible.getElements());
	    visibleBiasUpdatesKernel.setLearningRate(getLearningRate());
	    visibleBiasUpdatesKernel.setMomentum(getMomentum());
	    visibleBiasUpdatesKernel.setMiniBatchSize(posPhaseVisible.getColumns());
	    visibleBiasUpdatesKernel.setExecutionMode(Environment.getInstance().getExecutionMode());
	    visibleBiasUpdatesKernel.execute(rbm.getVisibleBiasConnections().getConnectionGraph().getElements().length);
	}

	// update hidden bias
	if (hiddenBiasUpdatesKernel != null) {
	    hiddenBiasUpdatesKernel.setPosPhase(posPhaseHidden.getElements());
	    hiddenBiasUpdatesKernel.setNegPhase(negPhaseHidden.getElements());
	    hiddenBiasUpdatesKernel.setLearningRate(getLearningRate());
	    hiddenBiasUpdatesKernel.setMomentum(getMomentum());
	    hiddenBiasUpdatesKernel.setMiniBatchSize(posPhaseHidden.getColumns());
	    hiddenBiasUpdatesKernel.setExecutionMode(Environment.getInstance().getExecutionMode());
	    hiddenBiasUpdatesKernel.execute(rbm.getHiddenBiasConnections().getConnectionGraph().getElements().length);
	}
    }

    protected void init() {
	RBM nn = getNeuralNetwork();

	if (!properties.containsKey(Constants.HIDDEN_CONNECTION_CALCULATOR)) {
	    properties.setParameter(Constants.HIDDEN_CONNECTION_CALCULATOR, new AparapiStochasticBinary());
	}

	if (!properties.containsKey(Constants.VISIBLE_CONNECTION_CALCULATOR)) {
	    properties.setParameter(Constants.VISIBLE_CONNECTION_CALCULATOR, new AparapiSigmoid());
	}

	if (weightUpdatesKernel == null) {
	    weightUpdatesKernel = new CDWeightUpdatesKernel(nn.getMainConnections().getConnectionGraph().getElements(), nn.getMainConnections().getConnectionGraph().getColumns());
	}

	if (visibleBiasUpdatesKernel == null && nn.getVisibleBiasConnections() != null) {
	    visibleBiasUpdatesKernel = new CDBiasUpdatesKernel(nn.getVisibleBiasConnections().getConnectionGraph().getElements());
	}

	if (hiddenBiasUpdatesKernel == null && nn.getHiddenBiasConnections() != null) {
	    hiddenBiasUpdatesKernel = new CDBiasUpdatesKernel(nn.getHiddenBiasConnections().getConnectionGraph().getElements());
	}
    }

    protected float getLearningRate() {
	return properties.getParameter(Constants.LEARNING_RATE);
    }
    
    protected float getMomentum() {
	return (float) (properties.getParameter(Constants.MOMENTUM) != null ? properties.getParameter(Constants.MOMENTUM) : 0f);
    }

    protected float getWeightDecay() {
	return (float) (properties.getParameter(Constants.WEIGHT_DECAY) != null ? properties.getParameter(Constants.WEIGHT_DECAY) : 0f);
    }
}
