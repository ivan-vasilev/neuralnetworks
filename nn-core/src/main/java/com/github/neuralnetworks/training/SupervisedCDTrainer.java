package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.util.Properties;

public abstract class SupervisedCDTrainer extends ContrastiveDivergenceTrainer {

    public SupervisedCDTrainer(Properties properties) {
	super(properties);
    }

    @Override
    protected Matrix getPositivePhaseVisible(TrainingInputData data) {
	if (posPhaseVisible == null || posPhaseVisible.getColumns() != data.getInput().getColumns()) {
	    posPhaseVisible = new Matrix(data.getInput().getRows() + data.getTarget().getRows(), data.getInput().getColumns());
	}

	System.arraycopy(data.getInput().getElements(), 0, posPhaseVisible, 0, data.getInput().getElements().length);
	System.arraycopy(data.getTarget().getElements(), 0, posPhaseVisible, data.getInput().getElements().length, data.getTarget().getElements().length);

	return posPhaseVisible;
    }
}
