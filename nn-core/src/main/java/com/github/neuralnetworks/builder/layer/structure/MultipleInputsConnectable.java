package com.github.neuralnetworks.builder.layer.structure;

/**
 * An interface for layer builder who can have more than one input.
 *
 * @author tmey
 */
public interface MultipleInputsConnectable
{

	/**
	 * Add one input layer into the <b>set</b> of inputs. So every name must be unique and not null or empty!
	 * 
	 * @param input
	 *          name of an input layer (unique, not empty, not null!)
	 */
	public void addInput(String input);
}
