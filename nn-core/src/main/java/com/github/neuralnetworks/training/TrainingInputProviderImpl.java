package com.github.neuralnetworks.training;

import java.util.ArrayList;
import java.util.List;

import com.github.neuralnetworks.calculation.neuronfunctions.MatrixFunction;
import com.github.neuralnetworks.input.InputModifier;

public abstract class TrainingInputProviderImpl implements TrainingInputProvider {

    /**
     * List of modifiers to apply on the input data after the conversion
     */
    private List<MatrixFunction> inputModifiers;

    @Override
    public TrainingInputData getNextInput() {
	TrainingInputData result = getNextUnmodifiedInput();
	if (result != null && inputModifiers != null) {
	    for (MatrixFunction m : inputModifiers) {
		m.value(result.getInput());
	    }
	}

	return result;
    }

    public void addInputModifier(MatrixFunction modifier) {
	if (inputModifiers == null) {
	    inputModifiers = new ArrayList<>();
	}

	inputModifiers.add(modifier);
    }

    public void removeModifier(InputModifier modifier) {
	if (inputModifiers != null) {
	    inputModifiers.remove(modifier);
	}
    }

    /**
     * @return base input without any input modifiers
     */
    protected abstract TrainingInputData getNextUnmodifiedInput();
}
