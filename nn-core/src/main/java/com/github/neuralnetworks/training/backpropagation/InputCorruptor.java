package com.github.neuralnetworks.training.backpropagation;

/**
 * Base interface for all methods of input corruption (denoising autoencoders, dropout etc)
 */
public interface InputCorruptor {
    public void corrupt(float[] values, float corruptionLevel);
}
