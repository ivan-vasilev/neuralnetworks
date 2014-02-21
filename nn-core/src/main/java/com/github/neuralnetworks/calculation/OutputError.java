package com.github.neuralnetworks.calculation;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * Implementations should calculate the output error, which will be presented as the result of the training. This is not the same as the output error derivative in BP.
 */
public interface OutputError {
    public void addItem(Matrix networkOutput, Matrix targetOutput);
    public float getTotalNetworkError();
    public int getTotalErrorSamples();
    public int getTotalInputSize();
    public void reset();
}
