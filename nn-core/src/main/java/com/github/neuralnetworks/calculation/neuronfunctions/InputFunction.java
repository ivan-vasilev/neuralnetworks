package com.github.neuralnetworks.calculation.neuronfunctions;

import java.io.Serializable;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Matrix;

/**
 * this interface is implemented by neuron input functions
 * 
 * @author hok
 * 
 */
public interface InputFunction extends Serializable {
    public void calculate(Connections graph, Matrix input, Matrix output);
}
