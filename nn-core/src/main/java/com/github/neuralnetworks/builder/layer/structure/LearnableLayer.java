package com.github.neuralnetworks.builder.layer.structure;

import com.github.neuralnetworks.training.random.RandomInitializer;

/**
 * @author tmey
 */
public interface LearnableLayer
{

	public void setLearningRate(float learningRate);

	public void setMomentum(float momentum);

	public void setL1weightDecay(float l1weightDecay);

	public void setL2weightDecay(float l2weightDecay);

	public RandomInitializer getWeightInitializer();

	public void setWeightInitializer(RandomInitializer weightInitializer);
}
