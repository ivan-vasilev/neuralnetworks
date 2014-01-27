package com.github.neuralnetworks.training.rbm;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;

/**
 * Base class for Aparapi Contrastive Divergence
 * Supports learning rate, momentum and weight decay
 */
public class AparapiCDTrainer extends CDTrainerBase {

    /**
     * weights update kernel for the connections between the visible and the hidden layer
     */
    private CDWeightUpdatesKernel weightUpdatesKernel;

    /**
     * weights update kernel for visible bias connections
     */
    private CDBiasUpdatesKernel visibleBiasUpdatesKernel;

    /**
     * weights update kernel for the hidden bias connections
     */
    private CDBiasUpdatesKernel hiddenBiasUpdatesKernel;

    public AparapiCDTrainer(Properties properties) {
	super(properties);
    }

    /* (non-Javadoc)
     * @see com.github.neuralnetworks.training.rbm.CDTrainerBase#updateWeights(com.github.neuralnetworks.architecture.Matrix, com.github.neuralnetworks.architecture.Matrix, com.github.neuralnetworks.architecture.Matrix, com.github.neuralnetworks.architecture.Matrix)
     * before each update the kernel update parameters are refreshed
     */
    @Override
    protected void updateWeights(Matrix posPhaseVisible, Matrix posPhaseHidden, Matrix negPhaseVisible, Matrix negPhaseHidden) {
	RBM rbm = getNeuralNetwork();

	int mbs = posPhaseHidden.getColumns();

	if (weightUpdatesKernel == null || weightUpdatesKernel.getMiniBatchSize() != mbs) {
	    weightUpdatesKernel = new CDWeightUpdatesKernel(rbm.getMainConnections().getConnectionGraph().getElements(), rbm.getMainConnections().getConnectionGraph().getColumns(), mbs);
	}

	weightUpdatesKernel.setPosPhaseVisible(posPhaseVisible.getElements());
	weightUpdatesKernel.setPosPhaseHidden(posPhaseHidden.getElements());
	weightUpdatesKernel.setNegPhaseVisible(negPhaseVisible.getElements());
	weightUpdatesKernel.setNegPhaseHidden(negPhaseHidden.getElements());
	weightUpdatesKernel.setLearningRate(getLearningRate());
	weightUpdatesKernel.setMomentum(getMomentum());
	weightUpdatesKernel.setWeightDecay(getWeightDecay());
	Environment.getInstance().getExecutionStrategy().execute(weightUpdatesKernel, rbm.getMainConnections().getConnectionGraph().getRows());

	// update visible bias
	if (rbm.getVisibleBiasConnections() != null) {
	    if (visibleBiasUpdatesKernel == null || visibleBiasUpdatesKernel.getMiniBatchSize() != mbs) {
		visibleBiasUpdatesKernel = new CDBiasUpdatesKernel(rbm.getVisibleBiasConnections().getConnectionGraph().getElements(), mbs);
	    }

	    visibleBiasUpdatesKernel.setPosPhase(posPhaseVisible.getElements());
	    visibleBiasUpdatesKernel.setNegPhase(negPhaseVisible.getElements());
	    visibleBiasUpdatesKernel.setLearningRate(getLearningRate());
	    visibleBiasUpdatesKernel.setMomentum(getMomentum());
	    Environment.getInstance().getExecutionStrategy().execute(visibleBiasUpdatesKernel, rbm.getVisibleBiasConnections().getConnectionGraph().getElements().length);
	}

	// update hidden bias
	if (rbm.getHiddenBiasConnections() != null) {
	    if (hiddenBiasUpdatesKernel == null || hiddenBiasUpdatesKernel.getMiniBatchSize() != mbs) {
		hiddenBiasUpdatesKernel = new CDBiasUpdatesKernel(rbm.getHiddenBiasConnections().getConnectionGraph().getElements(), mbs);
	    }

	    hiddenBiasUpdatesKernel.setPosPhase(posPhaseHidden.getElements());
	    hiddenBiasUpdatesKernel.setNegPhase(negPhaseHidden.getElements());
	    hiddenBiasUpdatesKernel.setLearningRate(getLearningRate());
	    hiddenBiasUpdatesKernel.setMomentum(getMomentum());
	    Environment.getInstance().getExecutionStrategy().execute(hiddenBiasUpdatesKernel, rbm.getHiddenBiasConnections().getConnectionGraph().getElements().length);
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
