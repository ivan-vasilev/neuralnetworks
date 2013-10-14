package com.github.neuralnetworks.input;

/**
 * 
 * implementations of this interface provide input for the netowrk
 * 
 */
public interface InputProvider {
    public float[] getNextInput();
}
