package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;

public interface BackPropagationConnectionCalculator extends ConnectionCalculator {

    public float getLearningRate();

    public void setLearningRate(float learningRate);

    public float getMomentum();

    public void setMomentum(float momentum);

    public float getL1weightDecay();

    public void setL1weightDecay(float l1weightDecay);
    
    public float getL2weightDecay();
    
    public void setL2weightDecay(float l2weightDecay);

    public ValuesProvider getActivations();

    public void setActivations(ValuesProvider activations);
}
