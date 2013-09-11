package com.github.neuralnetworks.architecture;

import com.github.neuralnetworks.architecture.activation.ActivationFunction;
import com.github.neuralnetworks.architecture.input.InputFunction;

/**
 *
 * this class represents a single neuron
 *
 * @author hok
 *
 */
public class Neuron {

	protected InputFunction inputFunction;
	protected ActivationFunction transferFunction;
	private IConnections inboundConnections;
	private IConnections outboundConnections;
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

	public NeuronConnections getInboundConnections() {
		return inboundConnections != null ? inboundConnections.getConnections(this) : null;
	}

	public NeuronConnections getOutboundConnections() {
		return outboundConnections != null ? outboundConnections.getConnections(this) : null;
	}

	/**
	 * this method "inserts" the neuron in the network
	 *
	 * @param inboundConnections
	 * @param outboundConnections
	 */
	public void insert(IConnections inboundConnections, IConnections outboundConnections) {
		if (inboundConnections != null && outboundConnections != null && inboundConnections.getOutputNeurons() != outboundConnections.getInputNeurons()) {
			throw new IllegalArgumentException("The neuron can be only in a single layer");
		}

		this.inboundConnections = inboundConnections;
		this.outboundConnections = outboundConnections;

		// retrieve the layer to "insert"
		Neuron[] layer = null;
		if (inboundConnections != null) {
			layer = inboundConnections.getOutputNeurons();
		} else if (outboundConnections != null) {
			layer = outboundConnections.getInputNeurons();
		}

		if (layer != null) {
			for (int i = 0; i < layer.length; i++) {
				if (layer[i] == null) {
					layer[i] = this;
					layerIndex = i;
					break;
				}
			}
		}
	}

	/**
	 * @return index of this neuron in the layer of neurons
	 */
	public Integer getLayerIndex() {
		if (layerIndex == null) {
			Neuron[] layer = inboundConnections.getOutputNeurons();
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
