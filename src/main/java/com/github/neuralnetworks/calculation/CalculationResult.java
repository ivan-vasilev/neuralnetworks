package com.github.neuralnetworks.calculation;

import java.io.Serializable;

import com.github.neuralnetworks.architecture.Neuron;

/**
 *
 * this class represents a calculation result on a layer of neurons based on the
 * result of the preceding layer
 *
 */
public class CalculationResult implements Serializable {

	private static final long serialVersionUID = -7948248396692132869L;

	private Neuron[] layer;
	private double[] calculation;

	public CalculationResult() {
		super();
	}

	public CalculationResult(Neuron[] layer, double[] calculation) {
		super();
		this.layer = layer;
		this.calculation = calculation;
	}

	public Neuron[] getLayer() {
		return layer;
	}

	public void setLayer(Neuron[] layer) {
		this.layer = layer;
	}

	public double[] getCalculation() {
		return calculation;
	}

	public void setCalculation(double[] calculation) {
		this.calculation = calculation;
	}

}
