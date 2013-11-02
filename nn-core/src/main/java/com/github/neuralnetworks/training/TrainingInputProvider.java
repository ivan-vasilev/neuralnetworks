package com.github.neuralnetworks.training;

/**
 * 
 * input provider for training data
 * 
 */
public interface TrainingInputProvider {
    public TrainingInputData getNextInput();
    public int getInputSize();
    public void reset();
}
