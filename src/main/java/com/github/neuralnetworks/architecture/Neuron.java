package com.github.neuralnetworks.architecture;

import java.util.Arrays;

import com.github.neuralnetworks.architecture.activation.ActivationFunction;
import com.github.neuralnetworks.architecture.input.InputFunction;

/**
 *
 * this class represents a single neuron
 *
 */
public class Neuron {

	protected InputFunction inputFunction;
	protected ActivationFunction transferFunction;
	private IConnections[] inboundConnectionGraphs;
	private IConnections[] outboundConnectionGraphs;
	private Neuron[] layer;
	private Integer layerIndex;

	public Neuron() {
		super();
	}

	public Neuron(InputFunction inputFunction, ActivationFunction transferFunction) {
		super();
		this.inputFunction = inputFunction;
		this.transferFunction = transferFunction;
	}

	public Neuron(IConnections inboundConnections, IConnections outboundConnections) {
		this(null, null, inboundConnections, outboundConnections);
	}

	public Neuron(InputFunction inputFunction, ActivationFunction transferFunction, IConnections inboundConnections, IConnections outboundConnections) {
		super();

		this.inputFunction = inputFunction;
		this.transferFunction = transferFunction;
		insert(inboundConnections, outboundConnections);
	}

	public InputFunction getInputFunction() {
		return inputFunction;
	}

	public void setInputFunction(InputFunction inputFunction) {
		this.inputFunction = inputFunction;
	}

	public ActivationFunction getTransferFunction() {
		return transferFunction;
	}

	public void setTransferFunction(ActivationFunction transferFunction) {
		this.transferFunction = transferFunction;
	}

	public NeuronConnections[] getInboundConnections() {
		NeuronConnections[] result = null;
		if (inboundConnectionGraphs != null && inboundConnectionGraphs.length > 0) {
			result = new NeuronConnections[inboundConnectionGraphs.length];
			for (int i = 0; i < inboundConnectionGraphs.length; i++) {
				result[i] = inboundConnectionGraphs[i].getConnections(this);
			}
		}

		return result;
	}

	public NeuronConnections[] getOutboundConnections() {
		NeuronConnections[] result = null;
		if (outboundConnectionGraphs != null && outboundConnectionGraphs.length > 0) {
			result = new NeuronConnections[outboundConnectionGraphs.length];
			for (int i = 0; i < outboundConnectionGraphs.length; i++) {
				result[i] = outboundConnectionGraphs[i].getConnections(this);
			}
		}

		return result;
	}

	/**
	 * this method "inserts" the neuron in the network
	 *
	 * @param inboundConnections
	 * @param outboundConnections
	 */
	public void insert(IConnections inbound, IConnections outbound) {
		if (inbound != null && outbound != null && inbound.getOutputNeurons() != outbound.getInputNeurons()) {
			throw new IllegalArgumentException("The neuron can be only in a single layer");
		}

		if (layer == null) {
			layer = inbound.getOutputNeurons();
		} else if ((inbound != null && inbound.getOutputNeurons() != layer) || (outbound != null && outbound.getInputNeurons() != layer)) {
			throw new IllegalArgumentException("The neuron can be only in a single layer");
		}

		if (inbound != null) {
			if (inboundConnectionGraphs == null) {
				inboundConnectionGraphs = new IConnections[1];
			} else {
				inboundConnectionGraphs = Arrays.copyOf(inboundConnectionGraphs, inboundConnectionGraphs.length + 1);
			}

			inboundConnectionGraphs[inboundConnectionGraphs.length - 1] = inbound;
		}

		if (outbound != null) {
			if (outboundConnectionGraphs == null) {
				outboundConnectionGraphs = new IConnections[1];
			} else {
				outboundConnectionGraphs = Arrays.copyOf(outboundConnectionGraphs, outboundConnectionGraphs.length + 1);
			}

			outboundConnectionGraphs[outboundConnectionGraphs.length - 1] = outbound;
		}

		if (outbound != null && outboundConnectionGraphs == null) {
			outboundConnectionGraphs = new IConnections[1];
			outboundConnectionGraphs[0] = outbound;
		}
	}

	/**
	 * @return index of this neuron in the layer of neurons
	 */
	public Integer getLayerIndex() {
		if (layerIndex == null && layer != null) {
			for (int i = 0; i < layer.length; i++) {
				if (layer[i] == this) {
					layerIndex = i;
					break;
				}
			}
		}

		return layerIndex;
	}
}
