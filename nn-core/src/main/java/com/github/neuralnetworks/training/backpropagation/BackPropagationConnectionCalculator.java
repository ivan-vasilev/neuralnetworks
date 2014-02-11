package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;

public interface BackpropagationConnectionCalculator extends ConnectionCalculator {

    public float getLearningRate();

    public void setLearningRate(float learningRate);

    public float getMomentum();

    public void setMomentum(float momentum);

    public float getWeightDecay();

    public void setWeightDecay(float weightDecay);

    public ValuesProvider getActivations();

    public void setActivations(ValuesProvider activations);
}
