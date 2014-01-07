package com.github.neuralnetworks.training.backpropagation;

import java.util.Map;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;

public interface BackpropagationConnectionCalculator extends ConnectionCalculator {

    public float getLearningRate();

    public void setLearningRate(float learningRate);

    public float getMomentum();

    public void setMomentum(float momentum);

    public float getWeightDecay();

    public void setWeightDecay(float weightDecay);

    public Map<Layer, Matrix> getActivations();

    public void setActivations(Map<Layer, Matrix> activations);
}
