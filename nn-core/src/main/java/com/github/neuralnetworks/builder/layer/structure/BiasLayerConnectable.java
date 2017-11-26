package com.github.neuralnetworks.builder.layer.structure;

import com.github.neuralnetworks.training.random.RandomInitializer;

/**
 * @author tmey
 */
public interface BiasLayerConnectable
{

	public boolean isAddBias();

	public void setAddBias(boolean addBias);

	public RandomInitializer getBiasWeightInitializer();

	public void setBiasWeightInitializer(RandomInitializer biasWeightInitializer);

	public void setBiasLearningRate(float biasLearningRate);

	public void setBiasMomentum(float biasMomentum);

	public void setBiasL1weightDecay(float biasL1weightDecay);

	public void setBiasL2weightDecay(float biasL2weightDecay);


}
