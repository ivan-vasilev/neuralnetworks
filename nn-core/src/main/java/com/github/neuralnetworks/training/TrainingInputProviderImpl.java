package com.github.neuralnetworks.training;

import java.util.ArrayList;
import java.util.List;

import com.github.neuralnetworks.calculation.neuronfunctions.TensorFunction;
import com.github.neuralnetworks.input.InputConverter;

public abstract class TrainingInputProviderImpl implements TrainingInputProvider {

    private static final long serialVersionUID = 1L;

    /**
     * List of modifiers to apply on the input data after the conversion
     */
    private List<TensorFunction> inputModifiers;

    /**
     * Converter for the target
     */
    private InputConverter inputConverter;

    /**
     * Counter
     */
    protected int currentInput;

    public TrainingInputProviderImpl() {
	super();
    }

    public TrainingInputProviderImpl(InputConverter inputConverter) {
	super();
	this.inputConverter = inputConverter;
    }

    @Override
    public List<TensorFunction> getInputModifiers() {
        return inputModifiers;
    }

    public void addInputModifier(TensorFunction modifier) {
	if (inputModifiers == null) {
	    inputModifiers = new ArrayList<>();
	}

	inputModifiers.add(modifier);
    }

    public void removeModifier(TensorFunction modifier) {
	if (inputModifiers != null) {
	    inputModifiers.remove(modifier);
	}
    }

    public InputConverter getInputConverter() {
        return inputConverter;
    }

    public void setInputConverter(InputConverter inputConverter) {
        this.inputConverter = inputConverter;
    }

    @Override
    public void beforeBatch(TrainingInputData ti) {
    }

    @Override
    public void afterBatch(TrainingInputData ti) {
    }

    @Override
    public void afterSample() {
	currentInput++;
    }

    @Override
    public void beforeSample() {
    }

    @Override
    public void reset() {
	currentInput = 0;
    }
}
