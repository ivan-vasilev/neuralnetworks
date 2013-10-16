package com.github.neuralnetworks.input;

import java.util.ArrayList;
import java.util.List;

import com.github.neuralnetworks.architecture.Matrix;

/**
 * 
 * Class for converting input values to matrices
 * 
 */
public abstract class InputConverter {

    private List<InputModifier> modifiers;
    private Matrix convertedInput;
    private Object[] notConvertedInput;

    public Matrix getConvertedInput(Object[] input) {
	if (notConvertedInput != input) {
	    notConvertedInput = input;
	    convertedInput = convert(input);
	    if (modifiers != null) {
		for (InputModifier m : modifiers) {
		    m.modify(convertedInput);
		}
	    }
	}

	return convertedInput;
    }

    public void addModifier(InputModifier modifier) {
	if (modifiers == null) {
	    modifiers = new ArrayList<>();
	}

	modifiers.add(modifier);
    }

    public void removeModifier(InputModifier modifier) {
	if (modifiers != null) {
	    modifiers.remove(modifier);
	}
    }

    protected abstract Matrix convert(Object[] input);
}
