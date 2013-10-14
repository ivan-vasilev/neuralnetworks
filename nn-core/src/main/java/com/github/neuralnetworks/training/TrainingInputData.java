package com.github.neuralnetworks.training;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.input.InputConverter;

/**
 * 
 * training input data with target value
 * 
 */
public class TrainingInputData {

    private Object[] input;
    private Object[] target;
    private InputConverter inputConverter;
    private InputConverter targetConverter;

    public TrainingInputData() {
	super();
    }

    public TrainingInputData(Object[] input) {
	super();
	this.input = input;
    }

    public TrainingInputData(Object[] input, Object[] target, InputConverter inputConverter, InputConverter targetConverter) {
	this.input = input;
	this.target = target;
	this.inputConverter = inputConverter;
	this.targetConverter = targetConverter;
    }

    public Object[] getTarget() {
	return target;
    }

    public void setTarget(Object[] target) {
	this.target = target;
    }

    public Object[] getInput() {
	return input;
    }

    public void setInput(Object[] input) {
	this.input = input;
    }

    public Matrix getConvertedInput() {
	return inputConverter.getConvertedInput(input);
    }

    public Matrix getConvertedTarget() {
	return targetConverter.getConvertedInput(target);
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
