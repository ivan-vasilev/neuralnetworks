package com.github.neuralnetworks.training;

import java.io.Serializable;

/**
 * Input provider for training data
 */
public interface TrainingInputProvider extends Serializable {
    public TrainingInputData getNextInput();
    public int getInputSize();
    public void reset();
}
