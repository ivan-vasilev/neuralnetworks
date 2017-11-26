package com.github.neuralnetworks.calculation;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.types.RBM;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.tensor.Tensor.TensorIterator;
import com.github.neuralnetworks.util.Environment;

/**
 * Implementation of LayerCalculatorImpl for RBMs
 * Contains some helper methods like calculateVisibleLayer and calculateHiddenLayer and also takes gibbs sampling into account
 */
public class RBMLayerCalculator implements Serializable
{

	private static final long serialVersionUID = 1L;

	private RBM rbm;
	private ValuesProvider posPhaseVP;
	private ValuesProvider negPhaseVP;
	private ConnectionCalculator posPhaseCC;
	private ConnectionCalculator negPhaseVisibleToHiddenCC;
	private ConnectionCalculator negPhaseHiddenToVisibleCC;

	public RBMLayerCalculator(RBM rbm, int miniBatchSize, ConnectionCalculator posPhaseCC, ConnectionCalculator negPhaseVisibleToHiddenCC, ConnectionCalculator negPhaseHiddenToVisibleCC)
	{
		super();
		this.rbm = rbm;
		this.posPhaseCC = posPhaseCC;
		this.negPhaseVisibleToHiddenCC = negPhaseVisibleToHiddenCC;
		this.negPhaseHiddenToVisibleCC = negPhaseHiddenToVisibleCC;
		this.posPhaseVP = TensorFactory.tensorProvider(rbm, miniBatchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());
		this.negPhaseVP = TensorFactory.tensorProvider(posPhaseVP, rbm);

		Tensor t = posPhaseVP.get(rbm.getVisibleBiasConnections().getInputLayer());
		TensorIterator it = t.iterator();
		while (it.hasNext()) {
			t.getElements()[it.next()] = 1;
		}

		t = posPhaseVP.get(rbm.getHiddenBiasConnections().getInputLayer());
		it = t.iterator();
		while (it.hasNext()) {
			t.getElements()[it.next()] = 1;
		}
		
		t = negPhaseVP.get(rbm.getVisibleBiasConnections().getInputLayer());
		it = t.iterator();
		while (it.hasNext()) {
			t.getElements()[it.next()] = 1;
		}
		
		t = negPhaseVP.get(rbm.getHiddenBiasConnections().getInputLayer());
		it = t.iterator();
		while (it.hasNext()) {
			t.getElements()[it.next()] = 1;
		}
	}

	public void gibbsSampling(RBM rbm, /* Matrix posPhaseVisible, Matrix posPhaseHidden, Matrix negPhaseVisible, Matrix negPhaseHidden, */int samplingCount, boolean resetNetwork)
	{
		List<Connections> connections = new ArrayList<>();
		connections.add(rbm.getMainConnections());
		if (rbm.getHiddenBiasConnections() != null)
		{
			connections.add(rbm.getHiddenBiasConnections());
		}
		TensorFactory.fill(posPhaseVP.get(rbm.getHiddenLayer()), 0);
		posPhaseCC.calculate(connections, posPhaseVP, rbm.getHiddenLayer());

		if (resetNetwork)
		{
			TensorFactory.copy(posPhaseVP.get(rbm.getHiddenLayer()), negPhaseVP.get(rbm.getHiddenLayer()));
		}

		// Gibbs sampling
		for (int i = 1; i <= samplingCount; i++)
		{
			connections.clear();
			connections.add(rbm.getMainConnections());
			if (rbm.getVisibleBiasConnections() != null)
			{
				connections.add(rbm.getVisibleBiasConnections());
			}
			TensorFactory.fill(negPhaseVP.get(rbm.getVisibleLayer()), 0);
			negPhaseHiddenToVisibleCC.calculate(connections, negPhaseVP, rbm.getVisibleLayer());

			connections.clear();
			connections.add(rbm.getMainConnections());
			if (rbm.getHiddenBiasConnections() != null)
			{
				connections.add(rbm.getHiddenBiasConnections());
			}
			TensorFactory.fill(negPhaseVP.get(rbm.getHiddenLayer()), 0);
			negPhaseVisibleToHiddenCC.calculate(connections, negPhaseVP, rbm.getHiddenLayer());
		}
	}

	public Matrix getPositivePhaseVisible()
	{
		return (Matrix) posPhaseVP.get(rbm.getVisibleLayer());
	}

	public Matrix getPositivePhaseHidden()
	{
		return (Matrix) posPhaseVP.get(rbm.getHiddenLayer());
	}

	public Matrix getNegativePhaseVisible()
	{
		return (Matrix) negPhaseVP.get(rbm.getVisibleLayer());
	}

	public Matrix getNegativePhaseHidden()
	{
		return (Matrix) negPhaseVP.get(rbm.getHiddenLayer());
	}

	public ConnectionCalculator getPosPhaseCC()
	{
		return posPhaseCC;
	}

	public ConnectionCalculator getNegPhaseVisibleToHiddenCC()
	{
		return negPhaseVisibleToHiddenCC;
	}

	public ConnectionCalculator getNegPhaseHiddenToVisibleCC()
	{
		return negPhaseHiddenToVisibleCC;
	}
}
