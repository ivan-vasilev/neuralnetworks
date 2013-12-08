package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.input.InputConverter;

/**
 * Training input data with target value default implementation
 */
public class TrainingInputDataImpl implements TrainingInputData {

    private Object[] realInput;
    private Object[] realTarget;

    /**
     * Converting input to Matrix
     */
    private InputConverter inputConverter;

    /**
     * Converting target to Matrix
     */
    private InputConverter targetConverter;

    public TrainingInputDataImpl() {
	super();
    }

    public TrainingInputDataImpl(Object[] realInput) {
	super();
	this.realInput = realInput;
    }

    public TrainingInputDataImpl(Object[] realInput, Object[] realTarget, InputConverter inputConverter, InputConverter targetConverter) {
	this.realInput = realInput;
	this.realTarget = realTarget;
	this.inputConverter = inputConverter;
	this.targetConverter = targetConverter;
    }

    public Object[] getRealTarget() {
	return realTarget;
    }

    public void setRealTarget(Object[] target) {
	this.realTarget = target;
    }

    public Object[] getRealInput() {
	return realInput;
    }

    public void setRealInput(Object[] realInput) {
	this.realInput = realInput;
    }

    @Override
    public Matrix getInput() {
	return inputConverter.getConvertedInput(realInput);
    }

    @Override
    public Matrix getTarget() {
	return targetConverter.getConvertedInput(realTarget);
    }

    public InputConverter getInputConverter() {
	return inputConverter;
    }

    public void setInputConverter(InputConverter inputConverter) {
	this.inputConverter = inputConverter;
    }

    public InputConverter getTargetConverter() {
        return targetConverter;
    }

    public void setTargetConverter(InputConverter targetConverter) {
        this.targetConverter = targetConverter;
    }
}
