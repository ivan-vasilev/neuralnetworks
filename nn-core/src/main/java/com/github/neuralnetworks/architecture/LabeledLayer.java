package com.github.neuralnetworks.architecture;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.Arrays;

import com.github.neuralnetworks.activation.ActivationFunction;
import com.github.neuralnetworks.neuroninput.InputFunction;

/**
 *
 * Layer with labeled neurons
 *
 */
public class LabeledLayer<L extends Serializable> extends Layer {

	private static final long serialVersionUID = 1L;

	private L[] labels;

	@SuppressWarnings("unchecked")
	public LabeledLayer(int neuronCount, InputFunction inputFunction, ActivationFunction activationFunction, Connections[] inboundConnectionGraphs, Connections[] outboundConnectionGraphs) {
		super(neuronCount, inputFunction, activationFunction, inboundConnectionGraphs, outboundConnectionGraphs);
		Class<L> c = null;
		this.labels = (L[]) Array.newInstance(c, neuronCount);
	}

	@SuppressWarnings("unchecked")
	public LabeledLayer(int neuronCount, InputFunction inputFunction, ActivationFunction activationFunction) {
		super(neuronCount, inputFunction, activationFunction);
		Class<L> c = null;
		this.labels = (L[]) Array.newInstance(c, neuronCount);
	}

	@SuppressWarnings("unchecked")
	public LabeledLayer(int neuronCount) {
		super(neuronCount);
		Class<L> c = null;
		this.labels = (L[]) Array.newInstance(c, neuronCount);
	}

	public L[] getLabels() {
		return labels;
	}

	public void setLabels(L[] labels) {
		this.labels = labels;
	}

	@Override
	public void setNeuronCount(int neuronCount) {
		super.setNeuronCount(neuronCount);
		if (labels.length != neuronCount) {
			labels = Arrays.copyOf(labels, neuronCount);
		}
	}
}
