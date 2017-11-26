package com.github.neuralnetworks.calculation.operations.aparapi.rbm;

import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.calculation.RBMLayerCalculator;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.training.rbm.CDTrainerBase;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;

/**
 * Base class for Aparapi Contrastive Divergence
 * Supports learning rate, momentum and weight decay
 */
public class AparapiCDTrainer extends CDTrainerBase
{

	private static final long serialVersionUID = 1L;

	/**
	 * weights update kernel for the connections between the visible and the hidden layer
	 */
	private CDWeightUpdatesKernel weightUpdatesKernel;

	/**
	 * weights update kernel for visible bias connections
	 */
	private CDBiasUpdatesKernel visibleBiasUpdatesKernel;

	/**
	 * weights update kernel for the hidden bias connections
	 */
	private CDBiasUpdatesKernel hiddenBiasUpdatesKernel;

	public AparapiCDTrainer(Properties properties)
	{
		super(properties);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.github.neuralnetworks.training.rbm.CDTrainerBase#updateWeights(com.github.neuralnetworks.architecture.Matrix, com.github.neuralnetworks.architecture.Matrix,
	 * com.github.neuralnetworks.architecture.Matrix, com.github.neuralnetworks.architecture.Matrix)
	 * before each update the kernel update parameters are refreshed
	 */
	@Override
	protected void updateWeights()
	{
		RBM rbm = getNeuralNetwork();

		RBMLayerCalculator lc = getLayerCalculator();
		int mbs = lc.getPositivePhaseVisible().getDimensions()[lc.getPositivePhaseVisible().getDimensions().length - 1];

		Hyperparameters hp = getHyperparameters();

		if (weightUpdatesKernel == null || weightUpdatesKernel.getMiniBatchSize() != mbs)
		{
			weightUpdatesKernel = new CDWeightUpdatesKernel(lc.getPositivePhaseVisible(), lc.getPositivePhaseHidden(), lc.getNegativePhaseVisible(), lc.getNegativePhaseHidden(), rbm.getMainConnections()
					.getWeights(), hp.getLearningRate(rbm.getMainConnections()), hp.getMomentum(rbm.getMainConnections()), hp.getL1WeightDecay(rbm.getMainConnections()), hp.getL2WeightDecay(rbm.getMainConnections()));
		}
		Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(weightUpdatesKernel, rbm.getMainConnections().getWeights().getRows());

		// update visible bias
		if (rbm.getVisibleBiasConnections() != null)
		{
			if (visibleBiasUpdatesKernel == null || visibleBiasUpdatesKernel.getMiniBatchSize() != mbs)
			{
				visibleBiasUpdatesKernel = new CDBiasUpdatesKernel(rbm.getVisibleBiasConnections().getWeights(), lc.getPositivePhaseVisible(), lc.getNegativePhaseVisible(), hp.getLearningRate(rbm.getVisibleBiasConnections()), hp.getMomentum(rbm.getVisibleBiasConnections()));
			}

			Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(visibleBiasUpdatesKernel, rbm.getVisibleBiasConnections().getWeights().getSize());
		}

		// update hidden bias
		if (rbm.getHiddenBiasConnections() != null)
		{
			if (hiddenBiasUpdatesKernel == null || hiddenBiasUpdatesKernel.getMiniBatchSize() != mbs)
			{
				hiddenBiasUpdatesKernel = new CDBiasUpdatesKernel(rbm.getHiddenBiasConnections().getWeights(), lc.getPositivePhaseHidden(), lc.getNegativePhaseHidden(), hp.getLearningRate(rbm.getHiddenBiasConnections()), hp.getMomentum(rbm.getHiddenBiasConnections()));
			}

			Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(hiddenBiasUpdatesKernel, rbm.getHiddenBiasConnections().getWeights().getSize());
		}
	}
}
