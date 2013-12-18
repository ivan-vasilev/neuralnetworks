package com.github.neuralnetworks.training.backpropagation;

import java.io.Serializable;

/**
 * Base interface for all methods of input corruption (denoising autoencoders, dropout etc)
 */
public interface InputCorruptor extends Serializable {
    public void corrupt(float[] values);
}
